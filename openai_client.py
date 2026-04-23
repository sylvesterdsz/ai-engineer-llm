from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from typing import Tuple, TypedDict, Literal, List

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

DEFAULT_MAX_TOKENS = 300  # Balanced for short structured responses (definitions + examples)

def get_openai_response(client: OpenAI, messages: List[Message]) -> Tuple[bool, str]:
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except RateLimitError:
        return False, "Error: Rate limit exceeded. Please try again later."

    except APIConnectionError:
        return False, "Error: Network issue communicating with OpenAI."

    except APIError as e:
        return False, f"OpenAI API error: {str(e)}"

    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

    if not response.choices:
        return False, "Error: Empty response from OpenAI."
    return True, response.choices[0].message.content
    