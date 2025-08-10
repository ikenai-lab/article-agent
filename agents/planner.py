import os
import ollama
import json
import re
import traceback
import sys
from typing import Dict, List, Optional, Union, Tuple 
from dataclasses import dataclass, field
from enum import Enum
from agents.ollama_token_counter import chat_with_token_counts 

# This setup assumes the script is in an 'agents' directory
# and the project root is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# This is imported just to get the list of tool names for the prompt
# A mock will be created in the `if __name__ == '__main__'` block for direct execution
try:
    from agents.tools import AVAILABLE_TOOLS
except ImportError:
    AVAILABLE_TOOLS = {} # Define a default if the module isn't found during import

class PlanComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"

@dataclass
class PlanStep:
    step_id: int
    tool: str
    query: str
    priority: int = 1
    depends_on: List[int] = field(default_factory=list)
    expected_output: str = ""
    # New fields for enhanced search quality
    search_keywords: List[str] = field(default_factory=list)
    time_sensitivity: str = "any"  # "recent", "historical", "any"
    query_alternatives: List[str] = field(default_factory=list)

@dataclass
class ResearchPlan:
    topic: str
    reasoning: str
    complexity: PlanComplexity
    estimated_time: str
    steps: List[PlanStep]
    success_criteria: List[str]
    fallback_strategies: List[str]
    suggested_outline: List[str] = field(default_factory=list)

class PlannerAgent:
    def __init__(self, model_name="granite3.3-ctx"):
        print("🤖 Initializing Enhanced Planner Agent...")
        self.model_name = model_name
        self.max_retries = 3
        self.tool_descriptions = self._get_tool_descriptions()
        # New: Tool-specific query optimization patterns
        self.tool_query_patterns = self._initialize_tool_patterns()

    def _get_tool_descriptions(self) -> Dict[str, str]:
        descriptions = {}
        # Ensure AVAILABLE_TOOLS is not empty before iterating
        if not AVAILABLE_TOOLS:
            return {}
        for tool_name, tool_obj in AVAILABLE_TOOLS.items():
            descriptions[tool_name] = getattr(tool_obj, 'description', f"Tool for {tool_name}")
        return descriptions

    def _initialize_tool_patterns(self) -> Dict[str, Dict]:
        """Initialize tool-specific query optimization patterns based on actual tool implementations."""
        return {
            "search_general_web": {
                "optimal_length": "3-6 words",
                "style": "conversational, news-focused",
                "avoid": ["overly technical jargon", "academic terminology", "site: operators"],
                "prefer": ["current events", "recent developments", "product names", "company names"],
                "notes": "Uses DDGS with trafilatura scraping - works best with mainstream topics"
            },
            "search_tech_blogs": {
                "optimal_length": "3-5 words",
                "style": "industry buzzwords, product names",
                "avoid": ["academic jargon", "overly broad terms"],
                "prefer": ["product names", "company names", "industry trends", "technology names"],
                "notes": "Searches curated tech sites (TechCrunch, The Verge, Wired, etc.) - use industry terminology",
                "target_sites": ["techcrunch.com", "theverge.com", "wired.com", "arstechnica.com", "venturebeat.com"]
            },
            "search_news": {
                "optimal_length": "3-6 words", 
                "style": "news headline style",
                "avoid": ["technical jargon", "academic terms"],
                "prefer": ["breaking news terms", "current events", "proper nouns", "location names"],
                "notes": "Searches major news outlets (Reuters, BBC, NYT, etc.) - use news-style language",
                "target_sites": ["reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com"]
            },
            "search_finance_news": {
                "optimal_length": "3-5 words",
                "style": "financial terminology, market focus",
                "avoid": ["casual language", "non-financial terms"],
                "prefer": ["earnings", "stock", "market", "revenue", "financial metrics", "company tickers"],
                "notes": "Searches financial outlets (Bloomberg, WSJ, CNBC, etc.) - use financial terminology",
                "target_sites": ["bloomberg.com", "wsj.com", "cnbc.com", "marketwatch.com"]
            },
            "search_youtube_transcripts": {
                "optimal_length": "2-4 words", 
                "style": "topic + speaker/event focused",
                "avoid": ["very long phrases", "complex queries"],
                "prefer": ["expert names", "conference names", "presentation titles", "simple topic terms"],
                "notes": "Uses Whisper for transcription - works best with educational/presentation content"
            },
            "search_arxiv": {
                "optimal_length": "4-8 words",
                "style": "academic, technical terminology",
                "avoid": ["casual language", "brand names"],
                "prefer": ["research terms", "methodology names", "academic concepts", "technical fields"],
                "notes": "ArXiv academic search - use precise scientific terminology"
            },
            "get_current_time": {
                "optimal_length": "0 words",
                "style": "no query needed",
                "avoid": ["any parameters"],
                "prefer": ["use when current date/time context is needed"],
                "notes": "Returns current IST time - no query required"
            }
        }

    def _generate_response(self, prompt: str, temperature: float = 0.1) -> Dict:
        """Helper function to generate a response from the Ollama model."""
        return chat_with_token_counts(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature}
        )

    def _extract_topic_entities(self, topic: str) -> Tuple[Dict[str, List[str]], Dict]:
        """Extract key entities, concepts, and search terms from the topic."""
        prompt = f"""
Analyze this research topic and extract key search entities:

Topic: "{topic}"

Extract and categorize the following as JSON:
{{
  "companies": ["list of company names mentioned"],
  "technologies": ["list of technologies, products, or technical concepts"],
  "people": ["list of notable people, experts, or researchers"],
  "concepts": ["list of key concepts or themes"],
  "time_indicators": ["list of temporal aspects like 'latest', 'recent', 'current'"],
  "industry_sectors": ["list of relevant industries or sectors"]
}}

Focus on extracting actual entities that would make good search terms. Return only the JSON object.
"""
        
        result_data = self._generate_response(prompt, temperature=0.0)
        response = result_data['response']
        
        token_usage = {
            'prompt_tokens': result_data.get('prompt_tokens', 0),
            'eval_tokens': result_data.get('eval_tokens', 0),
            'total_tokens': result_data.get('total_tokens', 0)
        }
        
        try:
            entities = self._extract_json_from_response(response)
            if entities is None:
                # Fallback to empty structure
                entities = {
                    "companies": [], "technologies": [], "people": [],
                    "concepts": [], "time_indicators": [], "industry_sectors": []
                }
            return entities, token_usage
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
            return {
                "companies": [], "technologies": [], "people": [],
                "concepts": [], "time_indicators": [], "industry_sectors": []
            }, token_usage

    def _optimize_query_for_tool(self, base_query: str, tool_name: str, entities: Dict[str, List[str]]) -> Tuple[str, List[str], List[str]]:
        """Optimize a query specifically for the given tool using extracted entities."""
        
        if tool_name not in self.tool_query_patterns:
            return base_query, [], []
        
        # This function appears to be a stub in the original code,
        # so we will return placeholder values. The main logic is in the prompt.
        optimized = base_query
        alternatives = []
        search_keywords = []

        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        search_keywords = list(set(all_entities[:10]))
        
        return optimized, search_keywords, alternatives

    def _determine_time_sensitivity(self, topic: str, entities: Dict[str, List[str]]) -> str:
        """Determine if the topic requires recent information."""
        recent_indicators = [
            "latest", "recent", "current", "new", "updated", "2024", "2025",
            "today", "now", "this year", "earnings", "announcement", "breaking"
        ]
        
        topic_lower = topic.lower()
        
        if any(indicator in topic_lower for indicator in recent_indicators) or entities.get("time_indicators"):
            return "recent"
        
        news_indicators = ["earnings", "report", "announcement", "launch", "release"]
        if any(indicator in topic_lower for indicator in news_indicators):
            return "recent"
        
        return "any"

    def _analyze_topic_complexity(self, topic: str) -> Tuple[PlanComplexity, Dict]:
        """Analyze the topic to determine research complexity."""
        prompt = f"""
Analyze this research topic and classify its complexity:

Topic: "{topic}"

Consider:
- How many different aspects need to be researched
- Technical depth required
- Number of stakeholders/perspectives involved
- Data analysis requirements

Respond with only one word: "simple", "moderate", or "comprehensive"
"""
        
        result_data = self._generate_response(prompt, temperature=0.0)
        response = result_data['response']
        complexity_str = response.strip().lower()
        
        token_usage = {
            'prompt_tokens': result_data.get('prompt_tokens', 0),
            'eval_tokens': result_data.get('eval_tokens', 0),
            'total_tokens': result_data.get('total_tokens', 0)
        }
        
        try:
            return PlanComplexity(complexity_str), token_usage
        except ValueError:
            return PlanComplexity.MODERATE, token_usage

    def _create_enhanced_prompt(self, topic: str, tool_names: List[str], complexity: PlanComplexity, entities: Dict[str, List[str]] = None) -> str:
        """Create an enhanced prompt with tool descriptions, examples, and entity awareness."""
        
        tool_info = []
        for tool in tool_names:
            description = self.tool_descriptions.get(tool, "General purpose tool")
            pattern = self.tool_query_patterns.get(tool, {})
            
            tool_info.append(f"- {tool}: {description}")
            if pattern:
                tool_info.append(f"   Best queries: {pattern.get('style', 'standard queries')}, {pattern.get('optimal_length', '3-5 words')}")
        
        tools_section = "\n".join(tool_info)
        
        complexity_guidance = {
            PlanComplexity.SIMPLE: "Create 2-3 focused research steps with highly targeted queries. Use 1-2 primary tools. Simple 3-section outline.",
            PlanComplexity.MODERATE: "Create 4-6 research steps with optimized queries covering main aspects. Use 2-3 complementary tools. 4-5 section outline.", 
            PlanComplexity.COMPREHENSIVE: "Create 6-10 detailed research steps with strategic multi-tool approach. Use 3-4 tools for cross-validation. 6-8 section outline."
        }
        
        entity_context = ""
        if entities:
            entity_parts = []
            for key, values in entities.items():
                if values:
                    entity_parts.append(f"{key}: {', '.join(values[:3])}")
            if entity_parts:
                entity_context = f"\n\nKEY ENTITIES IDENTIFIED:\n{chr(10).join(entity_parts)}\n"
        
        return f"""
You are an expert research strategist and content planner. Your goal is to design an efficient, logical sequence of research steps with OPTIMIZED SEARCH QUERIES that will yield high-quality, relevant results.

TOPIC: "{topic}"
COMPLEXITY LEVEL: {complexity.value} - {complexity_guidance[complexity]}{entity_context}

AVAILABLE TOOLS:
{tools_section}

SEARCH QUERY OPTIMIZATION GUIDELINES:
1. **Tool-Specific Optimization**: Each tool has different search mechanisms and content types
   - search_general_web: Uses DuckDuckGo + trafilatura scraping - optimize for mainstream content
   - search_tech_blogs: Searches specific tech sites - use industry terminology 
   - search_news: Searches major news outlets - use news headline style
   - search_finance_news: Searches financial sites - use financial terminology
   - search_youtube_transcripts: Uses Whisper transcription - simple, speaker-focused queries
   - search_arxiv: Academic search - use precise scientific terminology
   - get_current_time: No query needed - use for time context

2. **Query Length Optimization**: 
   - Web/News: 3-6 words (broader reach)
   - Tech Blogs: 3-5 words (focused industry terms)
   - YouTube: 2-4 words (simple, discoverable)
   - ArXiv: 4-8 words (precise academic terms)

3. **Entity-Based Optimization**: Use extracted entities strategically
   - Companies: Great for tech blogs, finance news, general web
   - People: Excellent for YouTube, news
   - Technologies: Perfect for tech blogs, arXiv
   - Time indicators: Critical for web and news searches

4. **Avoid Tool-Specific Pitfalls**:
   - Don't use site: operators in queries (tools handle this internally)
   - Don't use overly academic terms for YouTube/general web
   - Don't use casual language for arXiv
   - Consider that finance tools expect financial terminology

Create a research plan and suggested article outline following this EXACT JSON structure:

{{
  "reasoning": "Detailed explanation of your research strategy, tool selection rationale, and query optimization approach",
  "complexity": "{complexity.value}",
  "estimated_time": "Estimated completion time (e.g., '30-45 minutes')",
  "steps": [
    {{
      "step_id": 1,
      "tool": "tool_name_from_list",
      "query": "Optimized, targeted search query tailored for this specific tool",
      "priority": 1,
      "depends_on": [],
      "expected_output": "Brief description of what this step should provide",
      "search_keywords": ["key", "search", "terms"],
      "time_sensitivity": "recent|historical|any",
      "query_alternatives": ["alternative query 1", "alternative query 2"]
    }}
  ],
  "success_criteria": [
    "Specific criteria that define successful research completion"
  ],
  "fallback_strategies": [
    "Alternative approaches if primary tools fail, including query reformulation strategies"
  ],
  "suggested_outline": [
    "Introduction: [Brief description]",
    "Key Aspect 1: [Brief description]",
    "Key Aspect 2: [Brief description]",
    "Conclusion: [Brief description]"
  ]
}}

ENHANCED PLANNING GUIDELINES:
1. Start with broad context queries, then narrow to specifics
2. Optimize each query for its target tool's search algorithm and content type
3. Include 2-3 alternative queries for critical steps
4. Set time_sensitivity based on topic requirements
5. Extract relevant keywords that could be used for query refinement
6. Plan for cross-validation across multiple sources
7. Consider information recency and source authority requirements
8. Design queries to avoid common search pitfalls (too broad, too narrow, wrong terminology)

QUERY OPTIMIZATION EXAMPLES BY TOOL:
- search_general_web: "NVIDIA Q4 earnings AI industry" (mainstream, news-focused)
- search_tech_blogs: "NVIDIA AI chip analysis" (industry terminology)
- search_news: "NVIDIA earnings report impact" (news headline style)
- search_finance_news: "NVIDIA earnings revenue guidance" (financial metrics focus)
- search_youtube_transcripts: "Jensen Huang keynote" (person + event focused)  
- search_arxiv: "transformer architecture GPU optimization" (technical precision)
- get_current_time: "" (no query - provides current IST time for context)

Provide ONLY the JSON object. No additional text.
"""

    def _validate_and_clean_plan(self, plan_dict: Dict) -> Optional[ResearchPlan]:
        """Validate and convert the plan dictionary to a ResearchPlan object."""
        try:
            required_fields = ['reasoning', 'steps']
            for field_name in required_fields:
                if field_name not in plan_dict:
                    raise ValueError(f"Missing required field: {field_name}")
            
            steps = []
            for i, step_data in enumerate(plan_dict['steps']):
                step = PlanStep(
                    step_id=step_data.get('step_id', i + 1),
                    tool=step_data.get('tool', ''),
                    query=step_data.get('query', ''),
                    priority=step_data.get('priority', 2),
                    depends_on=step_data.get('depends_on', []),
                    expected_output=step_data.get('expected_output', ''),
                    search_keywords=step_data.get('search_keywords', []),
                    time_sensitivity=step_data.get('time_sensitivity', 'any'),
                    query_alternatives=step_data.get('query_alternatives', [])
                )
                steps.append(step)
            
            complexity_str = plan_dict.get('complexity', 'moderate')
            try:
                complexity = PlanComplexity(complexity_str)
            except ValueError:
                complexity = PlanComplexity.MODERATE
            
            suggested_outline = plan_dict.get('suggested_outline', [])
            if not isinstance(suggested_outline, list) or not all(isinstance(item, str) for item in suggested_outline):
                print("⚠️ Warning: 'suggested_outline' is not a list of strings. Defaulting to empty list.")
                suggested_outline = []

            return ResearchPlan(
                topic=plan_dict.get('topic', ''),
                reasoning=plan_dict['reasoning'],
                complexity=complexity,
                estimated_time=plan_dict.get('estimated_time', 'Not specified'),
                steps=steps,
                success_criteria=plan_dict.get('success_criteria', []),
                fallback_strategies=plan_dict.get('fallback_strategies', []),
                suggested_outline=suggested_outline
            )
            
        except Exception as e:
            print(f"❌ Plan validation failed: {e}")
            traceback.print_exc()
            return None

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from model response with multiple fallback strategies."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'\{.*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    # Fix trailing commas
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None

    def _refine_queries_post_generation(self, plan: ResearchPlan) -> ResearchPlan:
        """Post-process the plan to further refine queries based on tool patterns."""
        for step in plan.steps:
            if step.tool in self.tool_query_patterns:
                pattern = self.tool_query_patterns[step.tool]
                query_words_count = len(step.query.split())

                # Example refinement: shorten long general web queries
                if step.tool == "search_general_web" and query_words_count > 6:
                    step.query = ' '.join(step.query.split()[:5])
                
                # Example refinement: add context to short finance queries
                elif step.tool == "search_finance_news" and query_words_count < 3:
                     if not any(term in step.query.lower() for term in ["earnings", "stock", "market"]):
                         step.query += " earnings"

            # Add time modifier if needed
            if step.time_sensitivity == "recent" and step.tool in ["search_general_web", "search_news", "search_tech_blogs"]:
                 if not any(term in step.query.lower() for term in ["latest", "recent", "current", "breaking", "2024", "2025"]):
                     step.query = f"latest {step.query}"
        return plan

    def generate_plan(self, topic: str, tool_names: Optional[List[str]] = None, 
                      complexity_override: Optional[PlanComplexity] = None) -> Tuple[Optional[ResearchPlan], Dict]:
        """
        Generates an enhanced research plan for a given topic.
        Returns a tuple of (ResearchPlan, token_usage_dict).
        """
        print(f"🧠 Generating enhanced research plan for: '{topic}'...")
        total_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        if tool_names is None:
            tool_names = list(self.tool_descriptions.keys())
        
        print("   -> Extracting topic entities...")
        entities, entity_tokens = self._extract_topic_entities(topic)
        for key in total_tokens: total_tokens[key] += entity_tokens.get(key, 0)
        
        if complexity_override:
            complexity = complexity_override
        else:
            print("   -> Analyzing topic complexity...")
            complexity, complexity_tokens = self._analyze_topic_complexity(topic)
            for key in total_tokens: total_tokens[key] += complexity_tokens.get(key, 0)
        
        print(f"   -> Detected complexity: {complexity.value}")
        
        for attempt in range(self.max_retries):
            try:
                print(f"   -> Attempt {attempt + 1}/{self.max_retries} - Creating optimized research plan...")
                prompt = self._create_enhanced_prompt(topic, tool_names, complexity, entities)
                result_data = self._generate_response(prompt)
                response = result_data['response']

                for key in ['prompt_tokens', 'eval_tokens', 'total_tokens']: 
                    total_tokens[key] += result_data.get(key, 0)
                
                if not response: continue
                
                plan_dict = self._extract_json_from_response(response)
                if not plan_dict: continue
                
                plan_dict['topic'] = topic
                
                research_plan = self._validate_and_clean_plan(plan_dict)
                if research_plan:
                    print("   -> Applying post-generation query optimizations...")
                    research_plan = self._refine_queries_post_generation(research_plan)
                    
                    print("✅ Enhanced research plan generated successfully.")
                    return research_plan, total_tokens
                    
            except Exception as e:
                print(f"     ❌ Error on attempt {attempt + 1}: {e}")
        
        print("❌ Failed to generate research plan after all attempts.")
        return None, total_tokens

    def print_plan_summary(self, plan: ResearchPlan) -> None:
        """Print a formatted summary of the research plan."""
        print(f"\n{'='*60}")
        print(f"📋 RESEARCH PLAN: {plan.topic}")
        print(f"{'='*60}")
        print(f"🎯 Complexity: {plan.complexity.value.title()}")
        print(f"⏱️  Estimated Time: {plan.estimated_time}")
        print(f"\n💭 Strategy:\n   {plan.reasoning}")
        
        print(f"\n📝 Research Steps ({len(plan.steps)} total):")
        for step in plan.steps:
            priority_emoji = "🔴" if step.priority == 1 else "🟡" if step.priority == 2 else "🟢"
            time_emoji = "⏰" if step.time_sensitivity == "recent" else "📚" if step.time_sensitivity == "historical" else "🔍"
            depends_str = f" [Depends on: {', '.join(map(str, step.depends_on))}]" if step.depends_on else ""
            
            print(f"   {step.step_id}. {priority_emoji}{time_emoji} {step.tool}:")
            print(f"      Query: \"{step.query}\"{depends_str}")
            if step.query_alternatives:
                print(f"      Alternatives: {' | '.join(step.query_alternatives[:2])}")
            if step.expected_output:
                print(f"      Expected: {step.expected_output}")
            print()

        if plan.suggested_outline:
            print(f"✨ Suggested Article Outline:")
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

    def get_optimized_query_for_step(self, step: PlanStep, previous_results: Optional[Dict] = None) -> str:
        """
        Get an optimized query for a specific step at runtime.
        """
        base_query = step.query
        
        # This is where you could implement dynamic query refinement
        # based on what was found in earlier steps.
        if previous_results:
            pass # Placeholder for future implementation
        
        # Example runtime optimization: if a query is too long, use an alternative
        if step.query_alternatives and len(base_query.split()) > 6:
            return step.query_alternatives[0]
        
        return base_query
    
    def suggest_query_refinements(self, step: PlanStep, search_results: str) -> List[str]:
        """
        Analyze search results and suggest query refinements if results are poor.
        """
        suggestions = []
        
        if len(search_results.strip()) < 100:
            suggestions.append("Result is too short, broaden query by removing specific terms.")
            if step.query_alternatives:
                suggestions.append(f"Try alternative: '{step.query_alternatives[0]}'")
        
        # Add more logic here based on tool-specific failure modes
        if step.tool == "search_arxiv" and "No ArXiv papers found" in search_results:
            suggestions.append("Try broader academic terms or remove company names.")
        
        return suggestions


# This block is for testing the PlannerAgent directly.
if __name__ == '__main__':
    # For demonstration, we create a minimal mock for AVAILABLE_TOOLS
    # This prevents an ImportError if `agents.tools` doesn't exist.
    class MockTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description
        def __call__(self, *args, **kwargs): return "Mock content for testing."

    # If the real tools weren't imported, create mock ones.
    if not AVAILABLE_TOOLS:
        print("💡 Mocking AVAILABLE_TOOLS for local planner.py test run.")
        AVAILABLE_TOOLS = {
            "search_general_web": MockTool("search_general_web", "Search the general web for broad information."),
            "search_tech_blogs": MockTool("search_tech_blogs", "Search curated high-quality tech blogs."),
            "search_news": MockTool("search_news", "Search reputable news outlets."),
            "search_finance_news": MockTool("search_finance_news", "Search reputable financial news outlets."),
            "search_youtube_transcripts": MockTool("search_youtube_transcripts", "Search YouTube and get transcription data."),
            "search_arxiv": MockTool("search_arxiv", "Search arXiv for academic papers."),
            "get_current_time": MockTool("get_current_time", "Returns the current date and time."),
        }
    
    try:
        planner = PlannerAgent()
        
        tool_names = list(AVAILABLE_TOOLS.keys())
        
        TEST_TOPICS = [
            "The impact of NVIDIA's latest earnings report on the AI industry",
            "Climate change effects on global food security",
            "Latest developments in quantum computing hardware"
        ]
        
        for topic in TEST_TOPICS:
            print(f"\n🎯 Testing topic: {topic}")
            research_plan, token_usage = planner.generate_plan(topic, tool_names)
            
            if research_plan:
                planner.print_plan_summary(research_plan)
                print(f"📊 Token Usage: {token_usage['total_tokens']} total tokens")
                
                # Demo the runtime query optimization
                if research_plan.steps:
                    print(f"\n🔧 Runtime Query Optimization Demo:")
                    for step in research_plan.steps[:2]: # Show for first 2 steps
                        optimized = planner.get_optimized_query_for_step(step)
                        if optimized != step.query:
                            print(f"   Step {step.step_id}: '{step.query}' → '{optimized}'")
            else:
                print(f"\n❌ Failed to generate research plan for: {topic}")
                
    except Exception as e:
        print("\nAn unexpected error occurred during the test run:")
        traceback.print_exc()