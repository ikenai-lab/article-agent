import os
import google.generativeai as genai
import argparse
import time
import re
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Gemini API key.")
genai.configure(api_key=GOOGLE_API_KEY)


class GeminiMonolithicAgent:
    """
    A baseline agent that uses a single, large prompt to generate an article
    with a state-of-the-art model (Gemini Pro) for comparison purposes.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """
        Initializes the Gemini agent.

        Args:
            model_name (str): The name of the Gemini model to use.
        """
        print(f"🤖 Initializing Gemini Monolithic Agent with model: {model_name}")
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
        )

    def _create_monolithic_prompt(self, topic: str, style_sample_text: str, research_context: str, word_count: int) -> str:
        """
        Constructs a single, comprehensive prompt for the LLM.

        Args:
            topic (str): The article topic.
            style_sample_text (str): A text sample demonstrating the desired writing style.
            research_context (str): The combined research data to be used for facts.
            word_count (int): The target word count for the article.

        Returns:
            str: The fully constructed prompt.
        """
        return f"""
You are an expert author. Your task is to write a complete, high-quality article on a given topic. You must follow all instructions precisely.

**1. ARTICLE TOPIC:**
{topic}

**2. TARGET WORD COUNT:**
Approximately {word_count} words.

**3. WRITING STYLE INSTRUCTIONS:**
You must emulate the writing style of the following text sample. Pay close attention to its tone, voice, sentence structure, and vocabulary.

--- STYLE SAMPLE ---
{style_sample_text}
--- END STYLE SAMPLE ---

**4. FACTUAL CONTEXT:**
You MUST base all factual claims, data, and specific details in your article on the following research context. Do not introduce outside information.

--- RESEARCH CONTEXT ---
{research_context}
--- END RESEARCH CONTEXT ---

**5. TASK:**
Write a complete, well-structured article on the topic "{topic}".
- The article should have a clear introduction, body, and conclusion.
- Use headings and subheadings to structure the content logically.
- Ensure the final output is coherent, engaging, and strictly adheres to the provided writing style and factual context.

**FINAL ARTICLE OUTPUT:**
"""

    def generate_article(self, topic: str, style_sample_text: str, research_context: str, word_count: int) -> dict:
        """
        Generates an article using a single API call and measures performance.

        Args:
            topic (str): The article topic.
            style_sample_text (str): The style sample text.
            research_context (str): The combined research data.
            word_count (int): The target word count.

        Returns:
            dict: A dictionary containing the article, token counts, and execution time.
        """
        prompt = self._create_monolithic_prompt(topic, style_sample_text, research_context, word_count)
        
        print("\n--- Sending request to Gemini Pro ---")
        start_time = time.time()

        try:
            # Generate content
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Count tokens for the prompt
            prompt_token_count = self.model.count_tokens(prompt).total_tokens
            
            # Count tokens for the response
            # Note: Gemini API provides a way to count tokens for a given text.
            # We use the generated text to get the output token count.
            generated_text = response.text
            response_token_count = self.model.count_tokens(generated_text).total_tokens
            
            end_time = time.time()
            execution_time = end_time - start_time
            print("--- Response received ---")

            return {
                "article_text": generated_text,
                "prompt_tokens": prompt_token_count,
                "output_tokens": response_token_count,
                "total_tokens": prompt_token_count + response_token_count,
                "execution_time_seconds": execution_time
            }

        except Exception as e:
            print(f"❌ An error occurred during Gemini API call: {e}")
            return {
                "article_text": f"Error generating article: {e}",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "execution_time_seconds": time.time() - start_time
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A baseline script to generate an article using a single Gemini Pro prompt.")
    
    parser.add_argument('--topic', type=str, required=True, help="The topic of the article.")
    parser.add_argument('--style-file', type=str, required=True, help="Path to a .txt file with a writing style sample.")
    parser.add_argument('--context-file', type=str, default='data/combined_context.txt', help="Path to the combined research context file.")
    parser.add_argument('--target-word-count', type=int, default=1000, help="Target word count for the article.")

    args = parser.parse_args()

    # --- Read input files ---
    try:
        with open(args.style_file, 'r', encoding='utf-8') as f:
            style_text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Style file not found at '{args.style_file}'")
        exit(1)

    try:
        with open(args.context_file, 'r', encoding='utf-8') as f:
            context_text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Context file not found at '{args.context_file}'.")
        print("   Please run the main.py pipeline first to generate this file.")
        exit(1)

    # --- Run the agent ---
    agent = GeminiMonolithicAgent()
    result = agent.generate_article(
        topic=args.topic,
        style_sample_text=style_text,
        research_context=context_text,
        word_count=args.target_word_count
    )

    # --- Print results and save output ---
    print("\n\n" + "="*80)
    print("📊 GEMINI PRO BASELINE RESULTS")
    print("="*80)
    print(f"⏱️  Execution Time: {result['execution_time_seconds']:.2f} seconds")
    print(f"📝 Prompt Tokens: {result['prompt_tokens']}")
    print(f"📝 Output Tokens: {result['output_tokens']}")
    print(f"💰 Total Tokens: {result['total_tokens']}")
    print("="*80 + "\n")

    print("--- 📄 Generated Article ---")
    print(result["article_text"])
    print("---------------------------\n")

    # Save the generated article to the outputs directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    sanitized_topic = re.sub(r'[^\w\s-]', '', args.topic).replace(' ', '_').lower()
    filename = os.path.join(output_dir, f"{sanitized_topic}_gemini_baseline.md")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result["article_text"])
        
    print(f"✅ Article saved to '{filename}'")