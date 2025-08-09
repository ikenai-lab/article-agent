import json
import re
import ollama
from typing import Dict, Any, List

class StyleAnalyzerAgent:
    """
    A class to analyze the writing style of a given text using an LLM.
    It extracts stylistic features like tone, voice, sentence structure,
    vocabulary, and rhetorical devices.
    """

    def __init__(self, model_name: str = "gemma3", max_attempts: int = 3):
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
You are an expert literary analyst. Your task is to analyze the following text and generate a JSON object that describes its writing style in detail.

The JSON object must be valid and contain the following keys:
- "tone": Describe the overall tone (e.g., "Formal and academic", "Conversational and educational", "Technical and direct").
- "voice": Describe the narrative voice (e.g., "First-person narrative (uses 'I', 'my')", "Third-person objective").
- "sentence_structure": Analyze the complexity and variety of sentences (e.g., "Mix of simple and complex sentences", "Prefers short, declarative statements").
- "vocabulary": Describe the choice of words (e.g., "Mixes conversational language with technical terms", "Uses highly specialized jargon").
- "rhetorical_devices": Identify any specific literary or rhetorical techniques used (e.g., "Uses rhetorical questions", "Employs analogies to explain complex topics", "Uses lists to structure information").

IMPORTANT: Your response must be ONLY a valid JSON object. Do not include any text before or after the JSON. Do not use markdown formatting.
"""

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """
        Calls the Ollama chat API with the given prompts.

        Args:
            prompt (str): The user prompt containing the text to analyze.
            system_prompt (str): The system prompt defining the AI's role.

        Returns:
            str: The content of the response message from the model.
        """
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': system_prompt + "\n" + prompt}],
            options={"temperature": 0.0}
        )
        return response['message']['content']

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

        # Patterns to find JSON within markdown code blocks
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

    def analyze_style(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the writing style of the provided text.

        This method orchestrates the process of generating a prompt, calling the
        LLM, and parsing the JSON response. It will retry up to `max_attempts`
        if it fails to get a valid response.

        Args:
            text (str): The input text to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the style profile. Returns an
                            empty dictionary if a valid profile cannot be generated.
        """
        system_prompt = self._get_system_prompt()
        user_prompt = f"""
**TEXT TO ANALYZE:**
---
{text}
---
**JSON OUTPUT:**
"""
        for attempt in range(self.max_attempts):
            print(f"Analysis attempt {attempt + 1} of {self.max_attempts}...")
            try:
                raw_response = self._call_ollama(user_prompt, system_prompt)
                cleaned_json = self._extract_json(raw_response)

                if not cleaned_json:
                    print(f"Attempt {attempt + 1}: Could not extract JSON from response.")
                    continue

                result = json.loads(cleaned_json)

                if all(key in result for key in self.required_keys):
                    print("Successfully generated valid style profile.")
                    return result
                else:
                    missing_keys = [key for key in self.required_keys if key not in result]
                    print(f"Attempt {attempt + 1}: Missing required keys: {missing_keys}")

            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: Invalid JSON received.")
            except Exception as e:
                print(f"Attempt {attempt + 1}: An unexpected error occurred - {e}")

        print("Failed to produce a valid style profile after all attempts.")
        return {}

if __name__ == "__main__":
    try:
        # Ensure you have a file named 'style_sample.txt' in the same directory
        with open('./style_sample.txt', 'r', encoding='utf-8') as f:
            sample_text = f.read()
    except FileNotFoundError:
        print("Error: 'style_sample.txt' not found in the current directory.")
        print("Please create this file and add some text to it for analysis.")
        exit(1)

    # Instantiate the analyzer
    analyzer = StyleAnalyzerAgent(model_name="gemma3")

    # Get the style profile
    profile = analyzer.analyze_style(sample_text)

    # Print the result
    if profile:
        print("\n--- Generated Style Profile ---")
        print(json.dumps(profile, indent=2))
        print("-------------------------------")