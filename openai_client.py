import os
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

DEFAULT_MAX_TOKENS = 300  # Balanced for short structured responses (definitions + examples)

def get_openai_response(user_message: str) -> str:
    system_message = """You are a vocabulary specialist who helps users expand their English vocabulary.
                        When the user asks for a "word of the day", respond with:
                            - The word
                            - Its part of speech
                            - A clear, concise definition
                            - One or two example sentences
                            - A short tip or mnemonic to remember the word

                        Keep your explanations simple, engaging, and practical. Avoid overly technical language.
                        Avoid repeating very common words. Prefer slightly advanced vocabulary."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: API key not found. Check your .env file."
    client = OpenAI(api_key = api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except RateLimitError:
        return "Error: Rate limit exceeded. Please try again later."

    except APIConnectionError:
        return "Error: Network issue communicating with OpenAI."

    except APIError as e:
        return f"OpenAI API error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"

    if not response.choices:
        return "Error: Empty response from OpenAI."
    return response.choices[0].message.content
    