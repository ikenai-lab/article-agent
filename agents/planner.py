import os
import ollama
import json
import re
import traceback
import sys
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# This is imported just to get the list of tool names for the prompt
from agents.tools import AVAILABLE_TOOLS

class PlanComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"

@dataclass
class PlanStep:
    """Represents a single step in the research plan."""
    step_id: int # Added step_id as per LLM output expectation and dependencies
    tool: str
    query: str
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    depends_on: List[int] = field(default_factory=list) # Use default_factory for mutable defaults
    expected_output: str = ""  # What we expect to get from this step

@dataclass
class ResearchPlan:
    """Represents a complete research plan."""
    topic: str
    reasoning: str
    complexity: PlanComplexity
    estimated_time: str
    steps: List[PlanStep]
    success_criteria: List[str]
    fallback_strategies: List[str]
    suggested_outline: List[str] = field(default_factory=list) # NEW: Suggested high-level outline

class PlannerAgent:
    """
    An enhanced agent that analyzes a topic and creates a comprehensive research plan 
    by selecting the most appropriate tools and generating targeted queries with dependencies.
    It also suggests a high-level article outline.
    """
    
    def __init__(self, model_name="gemma3"):
        print("🤖 Initializing Enhanced Planner Agent...")
        self.model_name = model_name
        self.max_retries = 3
        # Dynamically get tool descriptions from the imported AVAILABLE_TOOLS
        self.tool_descriptions = self._get_tool_descriptions()
        
        # Check if the Ollama model is available
        try:
            print(f"  -> Checking for Ollama model: '{self.model_name}'...")
            ollama.show(self.model_name)
            print("✅ Ollama model found.")
        except Exception:
            print(f"❌ Ollama model '{self.model_name}' not found.")
            print(f"  Please follow the setup instructions to create it.")
            raise

    def _get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of available tools for better planning.
        Assumes AVAILABLE_TOOLS is a dictionary where values are objects with a 'description' attribute.
        """
        descriptions = {}
        for tool_name, tool_obj in AVAILABLE_TOOLS.items():
            # Use getattr to safely get the description, providing a default if not found
            descriptions[tool_name] = getattr(tool_obj, 'description', f"Tool for {tool_name}")
        return descriptions

    def _generate_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Helper function to generate a response from the Ollama model."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": temperature}
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error during Ollama generation: {e}")
            return ""

    def _analyze_topic_complexity(self, topic: str) -> PlanComplexity:
        """Analyze the topic to determine research complexity."""
        prompt = f"""
Analyze this research topic and classify its complexity:

Topic: "{topic}"

Consider factors like:
- How many different aspects need to be researched
- How recent/current the information needs to be
- How many different sources would be needed
- Technical complexity

Respond with only one word: "simple", "moderate", or "comprehensive"
"""
        
        response = self._generate_response(prompt, temperature=0.0)
        complexity_str = response.strip().lower()
        
        try:
            return PlanComplexity(complexity_str)
        except ValueError:
            return PlanComplexity.MODERATE  # Default fallback

    def _create_enhanced_prompt(self, topic: str, tool_names: List[str], complexity: PlanComplexity) -> str:
        """Create an enhanced prompt with tool descriptions and examples, including outline generation."""
        
        tool_info = []
        for tool in tool_names:
            description = self.tool_descriptions.get(tool, "General purpose tool")
            tool_info.append(f"- {tool}: {description}")
        
        tools_section = "\n".join(tool_info)
        
        complexity_guidance = {
            PlanComplexity.SIMPLE: "Create 2-3 focused research steps and a simple 3-section outline.",
            PlanComplexity.MODERATE: "Create 4-6 research steps covering main aspects and a 4-5 section outline.", 
            PlanComplexity.COMPREHENSIVE: "Create 6-10 detailed research steps with dependencies and a 6-8 section outline."
        }
        
        return f"""
You are an expert research strategist and content planner. Your goal is to design an efficient, logical sequence of research steps AND propose a high-level article outline.

TOPIC: "{topic}"
COMPLEXITY LEVEL: {complexity.value} - {complexity_guidance[complexity]}

AVAILABLE TOOLS:
{tools_section}

Create a research plan and a suggested article outline following this EXACT JSON structure:

{{
  "reasoning": "Detailed explanation of your research strategy and tool selection rationale",
  "complexity": "{complexity.value}",
  "estimated_time": "Estimated completion time (e.g., '30-45 minutes')",
  "steps": [
    {{
      "step_id": 1,
      "tool": "tool_name_from_list",
      "query": "Specific, targeted search query",
      "priority": 1,
      "depends_on": [],
      "expected_output": "Brief description of what this step should provide"
    }}
  ],
  "success_criteria": [
    "Specific criteria that define successful research completion"
  ],
  "fallback_strategies": [
    "Alternative approaches if primary tools fail"
  ],
  "suggested_outline": [
    "Introduction: [Brief description]",
    "Key Aspect 1: [Brief description]",
    "Key Aspect 2: [Brief description]",
    "Conclusion: [Brief description]"
  ]
}}

PLANNING GUIDELINES:
1. For 'steps', start with broad context, then narrow to specifics.
2. Consider information recency requirements for 'steps'.
3. Plan for data validation across multiple sources in 'steps'.
4. Include priority levels (1=critical, 2=important, 3=supplementary) for 'steps'.
5. Set dependencies where later steps need earlier results for 'steps'.
6. Design queries to be specific and actionable for 'steps'.
7. For 'suggested_outline', create a logical flow of high-level sections for an article on the TOPIC. Each item should be a string like "Section Title: Brief description". Ensure it matches the complexity guidance.

Provide ONLY the JSON object. No additional text.
"""

    def _validate_and_clean_plan(self, plan_dict: Dict) -> Optional[ResearchPlan]:
        """Validate and convert the plan dictionary to a ResearchPlan object."""
        try:
            # Validate required fields
            required_fields = ['reasoning', 'steps', 'suggested_outline'] # Added suggested_outline
            for field_name in required_fields:
                if field_name not in plan_dict:
                    raise ValueError(f"Missing required field: {field_name}")
            
            # Convert steps to PlanStep objects
            steps = []
            for i, step_data in enumerate(plan_dict['steps']):
                step = PlanStep(
                    step_id=step_data.get('step_id', i + 1), # Ensure step_id is present, default to index+1
                    tool=step_data.get('tool', ''),
                    query=step_data.get('query', ''),
                    priority=step_data.get('priority', 2),
                    depends_on=step_data.get('depends_on', []),
                    expected_output=step_data.get('expected_output', '')
                )
                steps.append(step)
            
            # Create ResearchPlan object
            complexity_str = plan_dict.get('complexity', 'moderate')
            try:
                complexity = PlanComplexity(complexity_str)
            except ValueError:
                complexity = PlanComplexity.MODERATE
            
            # Validate suggested_outline is a list of strings
            suggested_outline = plan_dict.get('suggested_outline', [])
            if not isinstance(suggested_outline, list) or not all(isinstance(item, str) for item in suggested_outline):
                print("⚠️ Warning: 'suggested_outline' is not a list of strings. Defaulting to empty list.")
                suggested_outline = []

            return ResearchPlan(
                topic=plan_dict.get('topic', ''), # 'topic' is added later in generate_plan, but good to have default
                reasoning=plan_dict['reasoning'],
                complexity=complexity,
                estimated_time=plan_dict.get('estimated_time', 'Not specified'),
                steps=steps,
                success_criteria=plan_dict.get('success_criteria', []),
                fallback_strategies=plan_dict.get('fallback_strategies', []),
                suggested_outline=suggested_outline # Pass the new field
            )
            
        except Exception as e:
            print(f"❌ Plan validation failed: {e}")
            traceback.print_exc() # Print traceback for validation errors
            return None

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from model response with multiple fallback strategies."""
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Regex extraction of JSON block
        json_patterns = [
            r'\{.*\}',  # Basic JSON object
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1) if match.lastindex == 1 else match.group(0) # Fix group selection
                try:
                    # FIX: Aggressively remove trailing commas before parsing
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try to fix common JSON issues
        try:
            # Remove common prefixes/suffixes
            cleaned = response.strip()
            for prefix in ['Here is the JSON:', 'JSON:', 'Plan:']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Find JSON boundaries more aggressively
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1 and end > start:
                potential_json = cleaned[start:end+1]
                # FIX: Aggressively remove trailing commas before parsing
                potential_json = re.sub(r',\s*([}\]])', r'\1', potential_json)
                return json.loads(potential_json)
                
        except json.JSONDecodeError:
            pass
        
        return None

    def generate_plan(self, topic: str, tool_names: Optional[List[str]] = None, 
                      complexity_override: Optional[PlanComplexity] = None) -> Optional[ResearchPlan]:
        """
        Generates an enhanced research plan for a given topic.

        Args:
            topic (str): The topic to research.
            tool_names (list, optional): Available tool names. Uses AVAILABLE_TOOLS.keys() if None.
            complexity_override (PlanComplexity, optional): Override automatic complexity detection.

        Returns:
            ResearchPlan: A comprehensive research plan object, or None if generation fails.
        """
        print(f"🧠 Generating enhanced research plan for: '{topic}'...")
        
        # Use provided tools or default to AVAILABLE_TOOLS keys
        if tool_names is None:
            tool_names = list(AVAILABLE_TOOLS.keys())
        
        # Determine complexity
        complexity = complexity_override or self._analyze_topic_complexity(topic)
        print(f"  -> Detected complexity: {complexity.value}")
        
        # Generate plan with retries
        for attempt in range(self.max_retries):
            try:
                print(f"  -> Attempt {attempt + 1}/{self.max_retries}")
                
                prompt = self._create_enhanced_prompt(topic, tool_names, complexity)
                response = self._generate_response(prompt)
                
                if not response:
                    print(f"    ❌ Empty response on attempt {attempt + 1}")
                    continue
                
                # Extract and parse JSON
                plan_dict = self._extract_json_from_response(response)
                if not plan_dict:
                    print(f"    ❌ Failed to extract JSON on attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        print(f"    Raw response (first 200 chars): {response[:200]}...")
                    continue
                
                # Add topic to plan dict (crucial for ResearchPlan dataclass)
                plan_dict['topic'] = topic
                
                # Validate and create plan object
                research_plan = self._validate_and_clean_plan(plan_dict)
                if research_plan:
                    print("✅ Enhanced research plan generated successfully.")
                    return research_plan
                else:
                    print(f"    ❌ Plan validation failed on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"    ❌ Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    traceback.print_exc() # Print full traceback on last attempt failure
        
        print("❌ Failed to generate research plan after all attempts.")
        return None

    def print_plan_summary(self, plan: ResearchPlan) -> None:
        """Print a formatted summary of the research plan."""
        print(f"\n{'='*60}")
        print(f"📋 RESEARCH PLAN: {plan.topic}")
        print(f"{'='*60}")
        print(f"🎯 Complexity: {plan.complexity.value.title()}")
        print(f"⏱️  Estimated Time: {plan.estimated_time}")
        print(f"\n💭 Strategy:")
        print(f"   {plan.reasoning}")
        
        print(f"\n📝 Research Steps ({len(plan.steps)} total):")
        for i, step in enumerate(plan.steps, 1):
            priority_emoji = "🔴" if step.priority == 1 else "🟡" if step.priority == 2 else "🟢"
            # Use step.step_id for dependencies display if available, otherwise use index
            depends_str = f" [Depends on: {', '.join(map(str, step.depends_on))}]" if step.depends_on else ""
            print(f"   {step.step_id}. {priority_emoji} {step.tool}: {step.query}{depends_str}")
            if step.expected_output:
                print(f"     Expected: {step.expected_output}")
        
        if plan.suggested_outline: # NEW: Print suggested outline
            print(f"\n✨ Suggested Article Outline:")
            for i, item in enumerate(plan.suggested_outline, 1):
                print(f"   {i}. {item}")

        if plan.success_criteria:
            print(f"\n✅ Success Criteria:")
            for criterion in plan.success_criteria:
                print(f"   • {criterion}")
        
        if plan.fallback_strategies:
            print(f"\n🔄 Fallback Strategies:")
            for strategy in plan.fallback_strategies:
                print(f"   • {strategy}")
        
        print(f"{'='*60}\n")

if __name__ == '__main__':
    # This block requires a dummy agents/tools.py for local testing if not already present
    # For demonstration, let's create a minimal mock for AVAILABLE_TOOLS
    class MockTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description
        def __call__(self, *args, **kwargs): return "Mock content for testing." # Dummy call method

    # Temporarily override or define AVAILABLE_TOOLS for testing in __main__
    # In a real setup, this would come directly from agents.tools
    if 'AVAILABLE_TOOLS' not in globals():
        print("💡 Mocking AVAILABLE_TOOLS for local planner.py test run.")
        sys.modules['agents.tools'] = type('module', (object,), {
            'AVAILABLE_TOOLS': {
                "search_general_web": MockTool("search_general_web", "Search the general web for broad information."),
                "search_youtube_transcripts": MockTool("search_youtube_transcripts", "Search YouTube for videos and get transcription data."),
                "search_arxiv": MockTool("search_arxiv", "Search arXiv for academic papers and preprints."),
                "search_tech_blogs": MockTool("search_tech_blogs", "Search curated high-quality tech blogs for industry insights.")
            }
        })
        from agents.tools import AVAILABLE_TOOLS
    
    try:
        planner = PlannerAgent()
        
        # Get the list of tool names from our tools module
        tool_names = list(AVAILABLE_TOOLS.keys())
        
        TEST_TOPICS = [
            "The impact of NVIDIA's latest earnings report on the AI industry",
            "Climate change effects on global food security",
            "Latest developments in quantum computing hardware"
        ]
        
        for topic in TEST_TOPICS:
            print(f"\n🎯 Testing topic: {topic}")
            research_plan = planner.generate_plan(topic, tool_names)
            
            if research_plan:
                planner.print_plan_summary(research_plan)
            else:
                print(f"\n❌ Failed to generate research plan for: {topic}")
                
    except Exception as e:
        print("\nAn unexpected error occurred during the test run:")
        traceback.print_exc()

