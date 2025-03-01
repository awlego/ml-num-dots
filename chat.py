import requests
import json
import ollama

def chat_with_ollama(prompt, model="deepseek-r1:7b"):
    """
    Send a prompt to Ollama and get the response with streaming
    
    Args:
        prompt (str): The input prompt to send
        model (str): The model to use (defaults to "deepseek-r1:7b")
    
    Returns:
        str: The model's complete response
    """
    messages = [{'role': 'user', 'content': prompt}]
    full_response = ""
    
    # Stream the response
    try:
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            if 'message' in chunk:
                content = chunk['message'].get('content', '')
                if content:  # Only process non-empty content
                    print(content, end='', flush=True)
                    full_response += content
            # Check if this is the final chunk
            if chunk.get('done', False):
                break
    except Exception as e:
        print(f"\nError during streaming: {e}")
    
    print()  # New line after streaming completes
    return full_response

if __name__ == "__main__":
    # Hardcoded prompt
    user_prompt = "How many '.' are in the following: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
    
    # Get response from Ollama
    response = chat_with_ollama(user_prompt)
    
    # Print the response
    print("\nOllama's response:")
    print(response)
