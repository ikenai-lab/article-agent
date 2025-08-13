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
    An agent that reviews a generated text against source context, persona, and cohesion.
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

    def critique_section(self, topic: str, section_title: str, draft_content: str, context: str, writing_style_json_str: str, article_context_summary: str, target_word_count: int) -> Dict:
        """
        Critiques a single section of the article based on a writer's persona.

        Args:
            topic (str): The overall topic of the article.
            section_title (str): The title of the section being critiqued.
            draft_content (str): The content of the draft section.
            context (str): The research context for this section (for factual consistency).
            writing_style_json_str (str): A JSON string describing the desired writer's persona.
            article_context_summary (str): A summary of the preceding article sections for coherence.
            target_word_count (int): The target word count for the section.

        Returns:
            dict: A dictionary containing 'critique', 'score', and token counts.
        """
        print(f"      -> Critiquing section: '{section_title}' for content and cohesion...")

        # Calculate word count and acceptable range
        actual_word_count = len(draft_content.split())
        word_count_tolerance = 0.25 # Allow 25% deviation
        min_words = int(target_word_count * (1 - word_count_tolerance))
        max_words = int(target_word_count * (1 + word_count_tolerance))

        # Prepare word count feedback
        word_count_feedback = ""
        if not (min_words <= actual_word_count <= max_words):
            word_count_feedback = f"4. **Word Count Adherence**: The draft has {actual_word_count} words, but the target is {target_word_count}. Please adjust the length to be between {min_words} and {max_words} words."
        
        # Check for gibberish at the end of the content
        gibberish_feedback = ""
        if re.search(r"(\s?,\s?[a-zA-Z]){5,}$", draft_content.strip()):
            gibberish_feedback = "5. **Completeness**: The section ends with garbled or incomplete text. The entire section must be rewritten to be a complete, coherent piece of text."

        prompt = f"""**Your Role:** You are a meticulous editor. Your task is to review a draft of a blog post section.

**Topic:** "{topic}"
**Section Title:** "{section_title}"

**Overall Article Context (Summary of what has already been written):**
---
{article_context_summary}
---

**Writer's Persona Profile (The draft MUST embody this persona):**
---
{writing_style_json_str}
---

**Research Context for This Section (for fact-checking):**
---
{context[:3000]}
---

**Draft to Review:**
---
{draft_content[:3000]}
---

**Your Task:**
Review the draft based on the following criteria and provide actionable feedback.
1.  **Persona Adherence:** Does the draft's tone, voice, and attitude match the writer's persona profile?
2.  **Factual Consistency:** Does the draft accurately represent the Research Context?
3.  **Cohesion & Flow:** Does the section logically follow the Overall Article Context without repetition?
{word_count_feedback}
{gibberish_feedback}

**Output Instructions:**
Provide ONLY a JSON object with two keys:
1.  "critique": A list of specific, actionable suggestions for improvement. If perfect, return an empty list.
2.  "score": An integer score from 1 (poor) to 5 (excellent). A score below 4 requires a rewrite.
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