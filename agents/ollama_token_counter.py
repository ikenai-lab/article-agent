# agents/llm_utils.py

import ollama
import re
from typing import Optional, Dict, Any

def chat_with_token_counts(
    model: str,
    prompt: str,
    system: str = "",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send a prompt to the model and return the response and token usage stats.
    
    Parameters:
        model (str): Name of the Ollama model to use.
        prompt (str): User message.
        system (str): Optional system prompt.
        options (dict): Ollama generation options (e.g., temperature, top_p, num_predict).
        
    Returns:
        dict: {
            'response': str,
            'prompt_tokens': int,
            'eval_tokens': int,
            'total_tokens': int
        }
    """
    messages = []
    
    if system:
        messages.append({'role': 'system', 'content': system})
    
    messages.append({'role': 'user', 'content': prompt})

    
    options = options or {}
    options.setdefault('num_predict', 8000)
    options.setdefault('stop', [])

    response = ollama.chat(
        model=model,
        messages=messages,
        options=options ,
        stream=False
    )

    prompt_tokens = response.get('prompt_eval_count', response.get('prompt_tokens', 0))
    eval_tokens = response.get('eval_count', response.get('eval_tokens', 0))
    cleaner_response = response.get('message', {}).get('content', '')
    # clean_response = re.sub(r'<think>.*?</think>\s*', '', cleaner_response, flags=re.DOTALL)
    return {
        'response': cleaner_response,
        'prompt_tokens': prompt_tokens,
        'eval_tokens': eval_tokens,
        'total_tokens': (prompt_tokens or 0) + (eval_tokens or 0)
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test chat_with_token_counts function")
    parser.add_argument(
        "--text", type=str, default=None,
        help="The text to send to the model. If not provided, a sample will be used.",
    )
    parser.add_argument(
        "--model", type=str, default="granite3.3-ctx",
        help="The Ollama model to use for testing."
    )
    args = parser.parse_args()

    response = chat_with_token_counts(args.model, args.text or "Hello, world!")
    print("Response:", response)