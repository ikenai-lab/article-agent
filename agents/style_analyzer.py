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
    A class to analyze the writing style of a given text using an LLM.
    It extracts stylistic features like tone, voice, sentence structure,
    vocabulary, and rhetorical devices.
    """

    def __init__(self, model_name: str = "granite3.3-ctx", max_attempts: int = 3):
        """
        Initializes the StyleAnalyzer.

        Args:
            model_name (str): The name of the Ollama model to use for analysis.
            max_attempts (int): The maximum number of times to try calling the API
                                and parsing the result.
        """
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.required_keys = ["tone", "voice", "sentence_structure", "vocabulary", "rhetorical_devices"]

    def _get_system_prompt(self) -> str:
        """
        Returns the static system prompt for the literary analyst expert.

        Returns:
            str: The system prompt text.
        """
        return """
You are a professional literary and linguistic style analyst.

Analyze ONLY the writing style of the given text and produce a single valid JSON object with exactly these keys:

- tone
- voice
- sentence_structure
- vocabulary
- rhetorical_devices

Output only the JSON object. No explanations or extra text.

CRITICAL: You are analyzing pure STYLE, not content. Imagine you can hear the rhythm and feel of the writing but cannot understand the meaning or topic. Focus exclusively on HOW the text is written, never WHAT it discusses.

Instructions for each key:

1. "tone": Describe the emotional quality and attitude using clear adjectives (e.g., conversational, authoritative, playful, serious, encouraging, skeptical). Focus purely on the writer's emotional stance and delivery style.

2. "voice": Describe the narrative perspective and writing persona (e.g., first-person confessional, third-person academic, personal mentor, distant observer, intimate guide). Describe the relationship between writer and reader, ignoring all subject matter.

3. "sentence_structure": Describe ONLY the mechanical and rhythmic patterns:
   - Length variation (short, long, mixed)
   - Complexity level (simple, compound, complex)
   - Pacing and flow (choppy, smooth, rhythmic, deliberate)
   - Structural rhythm and cadence
   ABSOLUTELY FORBIDDEN: Any mention of organizational methods, content types, formatting elements, or what the sentences contain.

4. "vocabulary": Describe ONLY the language register and accessibility level. Use ONLY these categories:
   - "Highly technical and specialized"
   - "Academic but accessible" 
   - "Conversational and informal"
   - "Formal and professional"
   - "Simple and straightforward"
   - "Complex and sophisticated"
   - "Mixed register with varied complexity"
   ABSOLUTELY FORBIDDEN: Mentioning any subject field, domain, discipline, or specific terminology types.

5. "rhetorical_devices": List ONLY standard rhetorical device names as a simple array. Never include examples, quotes, explanations, or content references. Use device names like: "Metaphor", "Direct address", "Rhetorical question", "Repetition", "Parallelism", "Analogy", "Alliteration", "Enumeration", "Personification", "Hyperbole".

Example output format:

{
  "tone": "Conversational and encouraging with slight self-deprecation",
  "voice": "First-person instructional, mentor-like and approachable",
  "sentence_structure": "Mixed lengths with moderate complexity and steady pacing",
  "vocabulary": "Accessible with moderate technical complexity",
  "rhetorical_devices": ["Direct address", "Enumeration", "Analogy"]
}

CONTENT-BLINDNESS RULES - NEVER VIOLATE:
- Do NOT identify subject fields, domains, topics, or disciplines
- Do NOT mention organizational methods (lists, bullet points, sections, etc.)
- Do NOT reference formatting or structural elements
- Do NOT quote or reference specific phrases from the text
- Do NOT list specific terms, jargon, or vocabulary examples
- Do NOT include parenthetical examples or explanations
- Do NOT mention what the text teaches, explains, discusses, or demonstrates

MENTAL MODEL: You are analyzing writing style like a music critic analyzing rhythm, tempo, and instrumentation without hearing the melody or understanding the lyrics. Focus on the technical craft of HOW language is used, not the meaning conveyed.
"""
    def _call_ollama(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """
        Calls the Ollama chat API with the given prompts using the centralized utility.

        Args:
            prompt (str): The user prompt containing the text to analyze.
            system_prompt (str): The system prompt defining the AI's role.

        Returns:
            Dict[str, Any]: A dictionary containing the response and token counts.
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

        Args:
            response (str): The string containing the JSON object.

        Returns:
            str: The extracted JSON string, or an empty string if not found.
        """
        if not isinstance(response, str):
            response = str(response)

        patterns: List[str] = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback to find the first '{' and last '}'
        try:
            start = response.index('{')
            end = response.rindex('}') + 1
            return response[start:end].strip()
        except ValueError:
            return ""

    def analyze_style(self, text: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Analyzes the writing style of the provided text.

        This method orchestrates the process of generating a prompt, calling the
        LLM, and parsing the JSON response. It will retry up to `max_attempts`
        if it fails to get a valid response.

        Args:
            text (str): The input text to analyze.

        Returns:
            Tuple containing:
            - A dictionary with the style profile.
            - A dictionary with token usage stats.
        """
        system_prompt = self._get_system_prompt()
        user_prompt= f"""
TEXT TO ANALYZE:
{text}
"""
        total_tokens = {'prompt_tokens': 0, 'eval_tokens': 0, 'total_tokens': 0}

        for attempt in range(self.max_attempts):
            print(f"Analysis attempt {attempt + 1} of {self.max_attempts}...")
            try:
                result_data = self._call_ollama(user_prompt, system_prompt)
                raw_response = result_data['response']
                print(f"Raw response: {raw_response}")  

                total_tokens['prompt_tokens'] += result_data.get('prompt_tokens', 0)
                total_tokens['eval_tokens'] += result_data.get('eval_tokens', 0)
                total_tokens['total_tokens'] += result_data.get('total_tokens', 0)

                cleaned_json = self._extract_json(raw_response)

                if not cleaned_json:
                    print(f"Attempt {attempt + 1}: Could not extract JSON from response.")
                    continue

                result = json.loads(cleaned_json)

                if all(key in result for key in self.required_keys):
                    print("Successfully generated valid style profile.")
                    return result, total_tokens
                else:
                    missing_keys = [key for key in self.required_keys if key not in result]
                    print(f"Attempt {attempt + 1}: Missing required keys: {missing_keys}")

            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: Invalid JSON received.")
            except Exception as e:
                print(f"Attempt {attempt + 1}: An unexpected error occurred - {e}")

        print("Failed to produce a valid style profile after all attempts.")
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
        print("\n--- Style Analysis Result ---")
        print(json.dumps(result, indent=4))
    else:
        print("\nFailed to analyze style.")

    print("\n--- Token Usage ---")
    print(json.dumps(token_stats, indent=4))


if __name__ == "__main__":
    main()
