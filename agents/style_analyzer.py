import json,sys,os
import re
from typing import Dict, Any, List, Tuple 
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from agents.ollama_token_counter import chat_with_token_counts

class StyleAnalyzerAgent:
    """
    A class to analyze a sample text and define a writer's persona.
    It extracts the author's role, attitude, voice, and target audience to create a holistic profile.
    """

    def __init__(self, model_name: str = "granite3.3-ctx", max_attempts: int = 3):
        """
        Initializes the StyleAnalyzer.

        Args:
            model_name (str): The name of the Ollama model to use for analysis.
            max_attempts (int): The maximum number of times to try calling the API.
        """
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.required_keys = ["author_persona", "tone_and_attitude", "writing_voice", "target_audience"]

    def _get_system_prompt(self) -> str:
        """
        Returns the static system prompt for the persona analysis expert.
        """
        return """
You are an expert literary analyst and profiler. Your task is to read a piece of text and create a detailed "persona profile" of the author.
This profile will be used by another AI to adopt the same writing style. Focus on capturing the *character* of the writer, not just technical details.

You must output a single, valid JSON object with exactly these four keys:
- "author_persona": Describe the writer's role or character. Are they a formal expert, an inquisitive journalist, a skeptical analyst, a passionate storyteller, a friendly guide? (e.g., "A curious and slightly informal investigator, making complex topics accessible.")
- "tone_and_attitude": Describe the emotional quality and attitude of the writing. Use descriptive adjectives. (e.g., "Inquisitive, grounded, slightly skeptical, and focused on clarity over jargon.")
- "writing_voice": Describe the writer's relationship with the reader and the structure of their writing. (e.g., "Uses first-person ('I') to guide the reader through a personal journey of discovery. Employs direct questions to engage the reader. Sentences are clear and direct, favoring narrative flow over dense academic structure.")
- "target_audience": Describe who the author is writing for. (e.g., "A general, intelligent audience who is curious about the topic but may not be an expert.")

CRITICAL: Do NOT analyze the content or topic. Focus exclusively on HOW the text is written and the persona it conveys.

Example output format:

{
  "author_persona": "An inquisitive journalist trying to make sense of a complex topic for the public.",
  "tone_and_attitude": "Skeptical but fair, grounded, and uses simple, direct language to build a narrative.",
  "writing_voice": "First-person perspective ('I'), guides the reader step-by-step, uses questions to frame the investigation.",
  "target_audience": "A general reader who is smart but not an expert in the specific domain."
}

Provide ONLY the JSON object. No explanations or extra text.
"""
    def _call_ollama(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """
        Calls the Ollama chat API with the given prompts using the centralized utility.
        """
        return chat_with_token_counts(
            model=self.model_name,
            prompt=prompt,
            system=system_prompt,
            options={"temperature": 0.0}
        )

    def _extract_json(self, response: str) -> str:
        """
        Extracts a JSON string from a larger string, handling markdown code blocks.
        """
        if not isinstance(response, str):
            response = str(response)

        # Find the first '{' and the last '}' to isolate the JSON block
        try:
            start = response.index('{')
            end = response.rindex('}') + 1
            return response[start:end].strip()
        except ValueError:
            # Fallback for markdown code blocks if the primary method fails
            patterns: List[str] = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```'
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    return match.group(1).strip()
            return ""

    def analyze_style(self, text: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Analyzes the writing style of the provided text to create a persona profile.

        Args:
            text (str): The input text to analyze.

        Returns:
            A tuple containing the style profile (persona) and token usage stats.
        """
        system_prompt = self._get_system_prompt()
        user_prompt= f"""
Please analyze the following text and generate a writer persona profile based on it.

TEXT TO ANALYZE:
---
{text}
---
"""
        total_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        for attempt in range(self.max_attempts):
            print(f"Persona analysis attempt {attempt + 1} of {self.max_attempts}...")
            try:
                result_data = self._call_ollama(user_prompt, system_prompt)
                raw_response = result_data['response']
                
                total_tokens['prompt_tokens'] += result_data.get('prompt_tokens', 0)
                total_tokens['eval_tokens'] += result_data.get('eval_tokens', 0)
                total_tokens['total_tokens'] += result_data.get('total_tokens', 0)

                cleaned_json_str = self._extract_json(raw_response)

                if not cleaned_json_str:
                    print(f"Attempt {attempt + 1}: Could not extract JSON from response.")
                    continue

                # Clean up potential trailing commas before parsing
                cleaned_json_str = re.sub(r',\s*([}\]])', r'\1', cleaned_json_str)
                result = json.loads(cleaned_json_str)

                if all(key in result for key in self.required_keys):
                    print("✅ Successfully generated valid persona profile.")
                    return result, total_tokens
                else:
                    missing_keys = [key for key in self.required_keys if key not in result]
                    print(f"Attempt {attempt + 1}: Missing required keys: {missing_keys}")

            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: Invalid JSON received. Error: {e}")
                print(f"Raw Response causing error: {raw_response}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: An unexpected error occurred - {e}")

        print("❌ Failed to produce a valid persona profile after all attempts.")
        return {}, total_tokens
    
def main():
    
    parser = argparse.ArgumentParser(description="Test the StyleAnalyzerAgent.")
    parser.add_argument(
        "--text",
        type=str,
        help="The text to analyze. If not provided, a sample will be used.",
    )
    args = parser.parse_args()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_file_path = os.path.join(root_dir, "style_sample.txt")
    with open(sample_file_path, "r") as f:
        style_sample_text = f.read().strip()
    

    input_text = args.text if args.text else style_sample_text

    analyzer = StyleAnalyzerAgent(model_name="granite3.3-ctx")
    result, token_stats = analyzer.analyze_style(input_text)

    if result:
        print("\n--- Persona Analysis Result ---")
        print(json.dumps(result, indent=4))
    else:
        print("\nFailed to analyze style.")

    print("\n--- Token Usage ---")
    print(json.dumps(token_stats, indent=4))


if __name__ == "__main__":
    main() 