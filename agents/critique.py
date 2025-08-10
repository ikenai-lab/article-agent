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
    sys.path.insert(0, project_root)

# Correctly import the utility that includes token counting
from agents.ollama_token_counter import chat_with_token_counts

class CritiqueAgent:
    """
    An agent that reviews a generated text against source context, style, and cohesion.
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

    def critique_section(self, topic: str, section_title: str, draft_content: str, context: str, writing_style_json_str: str, article_context_summary: str) -> Dict:
        """
        Critiques a single section of the article, including article-wide context.

        Args:
            topic (str): The overall topic of the article.
            section_title (str): The title of the section being critiqued.
            draft_content (str): The content of the draft section.
            context (str): The research context for this section (for factual consistency).
            writing_style_json_str (str): A JSON string describing the desired writing style.
            article_context_summary (str): A summary of the preceding article sections for coherence.

        Returns:
            dict: A dictionary containing 'critique', 'score', and token counts.
        """
        print(f"      -> Critiquing section: '{section_title}' for content and cohesion...")

        prompt = f"""**Your Role:** You are a meticulous editor. Your task is to review a draft of a blog post section.

**Topic:** "{topic}"
**Section Title:** "{section_title}"

**Overall Article Context (Summary of what has already been written):**
---
{article_context_summary}
---

**Research Context for This Section (for fact-checking):**
---
{context[:3000]}
---

**Writing Style Profile (to emulate):**
---
{writing_style_json_str}
---

**Draft to Review:**
---
{draft_content[:3000]}
---

**Your Task:**
Review the draft based on THREE primary criteria and provide actionable feedback:
1.  **Factual Consistency & Completeness:** Check if the draft accurately represents the Research Context.
2.  **Style Adherence:** Check if the draft's tone, voice, and structure match the Writing Style Profile.
3.  **Cohesion & Integration:** Check if the section logically follows the Overall Article Context without repetition.

**Output Instructions:**
Provide ONLY a JSON object with two keys:
1.  "critique": A list of specific, actionable suggestions for improvement. If perfect, return an empty list.
2.  "score": An integer score from 1 (poor) to 5 (excellent).

"""
        
        result_data = chat_with_token_counts(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": 0.0}
        )
        response_json_str = result_data['response']

        # Initialize the return dictionary, including the token counts from the start.
        final_result = {
            "critique": ["Failed to generate valid critique."], "score": 1,
            'prompt_tokens': result_data.get('prompt_tokens', 0),
            'eval_tokens': result_data.get('eval_tokens', 0),
            'total_tokens': result_data.get('total_tokens', 0)
        }

        if not response_json_str:
            print("      -> ❌ Empty response from critique model.")
            final_result["critique"] = ["Empty response from critique model."]
            return final_result

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
            # Update the dictionary with the parsed critique and score, preserving token counts.
            final_result.update(critique_plan)
            return final_result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"      -> ❌ Failed to parse or validate the critique: {e}")
            traceback.print_exc()
            return final_result