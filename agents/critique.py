import os
import ollama
import json
import re
import traceback
import sys
from typing import Dict

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.sys.path.insert(0, project_root)

from agents.ollama_token_counter import chat_with_token_counts

class CritiqueAgent:
    """
    An agent that reviews a generated text against source context and a style sample.
    """
    def __init__(self, model_name="granite3.3-ctx"):
        print("🧐 Initializing Critique Agent...")
        self.model_name = model_name
        try:
            ollama.show(self.model_name)
            print(f"✅ Ollama model '{self.model_name}' found for CritiqueAgent.")
        except Exception:
            print(f"❌ Ollama model '{self.model_name}' not found for CritiqueAgent.")
            print(f"  Please follow the setup instructions to create it.")
            raise

    def _generate_response(self, prompt: str) -> str:
        """Helper function to generate a response from the Ollama model."""
        try:
            response = chat_with_token_counts(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.0} # Low temperature for deterministic critique
            )
            return response
        except Exception as e:
            print(f"Error during Ollama generation for critique: {e}")
            traceback.print_exc() # Print traceback for debugging
            return ""

    def critique_section(self, topic: str, section_title: str, draft_content: str, context: str, writing_style_json_str: str) -> Dict:
        """
        Critiques a single section of the article.

        Args:
            topic (str): The overall topic of the article.
            section_title (str): The title of the section being critiqued.
            draft_content (str): The content of the draft section.
            context (str): The research context used to generate the draft (for factual consistency).
            writing_style_json_str (str): A JSON string describing the desired writing style.

        Returns:
            dict: A dictionary containing 'critique' (list of suggestions), 'score' (1-5), and token counts.
        """
        print(f"      -> Critiquing section: '{section_title}'...")
        
        context_limited = context[:3000]
        draft_content_limited = draft_content[:3000]

        prompt = f"""**Your Role:** You are a meticulous editor. Your task is to review a draft of a blog post section.

**Topic:** "{topic}"
**Section Title:** "{section_title}"

**Source Context (for fact-checking and completeness):**
---
{context_limited}
---

**Writing Style Profile (to emulate):**
---
{writing_style_json_str}
---

**Draft to Review:**
---
{draft_content_limited}
---

**Your Task:**
Review the draft based on two primary criteria and provide actionable feedback:
1.  **Factual Consistency & Completeness:**
    * Does the draft accurately represent the information in the Source Context?
    * Are there any claims in the draft not supported by the context?
    * Are there important facts or key points from the context that were missed in the draft?
2.  **Style Adherence:**
    * Does the draft's tone, voice, sentence structure, vocabulary level, rhetorical devices, paragraph style, and engagement techniques match the Writing Style Profile?
    * Are there any grammatical errors, typos, or awkward phrasing?
    * Is the word count roughly aligned with expectations (consider the section's target word count if known)?

**Output Instructions:**
Provide ONLY a JSON object with two keys:
1.  "critique": A list of specific, actionable suggestions for improvement. Each suggestion should be a concise string. If the draft is perfect (no improvements needed for factual consistency, completeness, or style adherence), return an empty list.
2.  "score": An integer score from 1 (poor, major issues) to 5 (excellent, no issues). Score 5 ONLY if the "critique" list is empty. Otherwise, score based on severity of issues (1-4).
"""
        
        result_data = chat_with_token_counts(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": 0.0}
        )
        response_json_str = result_data['response']

        base_return = {
            "critique": ["Failed to generate valid critique."], "score": 1,
            'prompt_tokens': result_data.get('prompt_tokens', 0),
            'eval_tokens': result_data.get('eval_tokens', 0),
            'total_tokens': result_data.get('total_tokens', 0)
        }

        if not response_json_str:
            print("      -> ❌ Empty response from critique model.")
            base_return["critique"] = ["Empty response from critique model."]
            return base_return

        try:
            json_match = re.search(r'\{.*?\}', response_json_str, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in the response.", response_json_str, 0)
            
            clean_json_str = json_match.group(0)
            clean_json_str = re.sub(r',\s*([}\]])', r'\1', clean_json_str)
            
            critique_plan = json.loads(clean_json_str)

            if not isinstance(critique_plan, dict) or "critique" not in critique_plan or "score" not in critique_plan:
                raise ValueError("Parsed JSON does not contain expected 'critique' or 'score' keys.")
            
            print("      -> Critique complete.")
            base_return.update(critique_plan)
            return base_return
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"      -> ❌ Failed to parse or validate the critique: {e}")
            traceback.print_exc()
            return base_return

if __name__ == '__main__':
    print("--- Testing CritiqueAgent ---")
    
    # Dummy data for testing
    test_topic = "The Impact of AI on Education"
    test_section_title = "Personalized Learning with AI"
    test_context = """
    AI can analyze student performance data to identify strengths and weaknesses. 
    Adaptive learning platforms powered by AI can then tailor content and pace to individual students. 
    Studies show this can improve engagement and learning outcomes. 
    However, concerns exist about data privacy and algorithmic bias in educational settings.
    """
    test_style_profile = json.dumps({
        "tone": "informative and objective",
        "voice": "third person, authoritative",
        "sentence_structure": "varied, leaning towards clear and concise",
        "vocabulary_level": "professional and accessible",
        "rhetorical_devices": "factual presentation, logical arguments",
        "paragraph_style": "well-structured, topic-sentence driven",
        "engagement_techniques": "clear explanations, data-driven insights"
    })

    # Test Case 1: Good draft
    good_draft = """
    Artificial intelligence is revolutionizing education by enabling personalized learning experiences. 
    AI-powered adaptive platforms analyze individual student data, pinpointing areas where they excel or struggle. 
    This allows for customized content delivery and pacing, significantly boosting student engagement and academic outcomes. 
    For instance, a platform might provide additional practice problems in algebra for a student who consistently misses those concepts, while accelerating another through material they've mastered.
    """
    print("\n--- Test Case 1: Good Draft ---")
    critique_agent = CritiqueAgent()
    result_good = critique_agent.critique_section(test_topic, test_section_title, good_draft, test_context, test_style_profile)
    print(f"Critique Result (Good): {result_good}")

    # Test Case 2: Draft with factual inconsistency/missing info
    bad_fact_draft = """
    AI is widely used in education, and it always ensures perfect data privacy. 
    Adaptive learning platforms are becoming common, but their impact on student engagement is still unclear.
    """
    print("\n--- Test Case 2: Bad Fact/Missing Info Draft ---")
    result_bad_fact = critique_agent.critique_section(test_topic, test_section_title, bad_fact_draft, test_context, test_style_profile)
    print(f"Critique Result (Bad Fact): {result_bad_fact}")

    # Test Case 3: Draft with style issues (e.g., informal tone)
    bad_style_draft = """
    So, like, AI in schools is super cool. It helps kids learn stuff better by, you know, giving them personalized lessons. 
    It's gonna change everything, for sure!
    """
    print("\n--- Test Case 3: Bad Style Draft ---")
    result_bad_style = critique_agent.critique_section(test_topic, test_section_title, bad_style_draft, test_context, test_style_profile)
    print(f"Critique Result (Bad Style): {result_bad_style}")

    # Test Case 4: Empty draft
    empty_draft = ""
    print("\n--- Test Case 4: Empty Draft ---")
    result_empty = critique_agent.critique_section(test_topic, test_section_title, empty_draft, test_context, test_style_profile)
    print(f"Critique Result (Empty): {result_empty}")

