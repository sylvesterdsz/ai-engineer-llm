from dotenv import load_dotenv
import os
from openai_client import get_openai_response
from embedding import get_embedding, cosine_similarity
from openai import OpenAI
from typing import List
from openai_client import Message
import sys

load_dotenv()

def main():
    system_message = """You are a vocabulary specialist who helps users expand their English vocabulary.

You handle vocabulary-related requests, including:
- Requests for new words (e.g., "word of the day", "give me a word")
- Definitions of words
- Differences between words (e.g., "affect vs effect")
- Usage, examples, or clarification of meaning
- Etymology or word origins

If a request is not related to vocabulary or words, politely decline and state that you can only assist with vocabulary-related queries.

If the user asks for a new word (e.g., "word of the day", "give me a word", "new vocabulary", or similar), respond using the exact structured format below:

Word: <word>
Part of Speech: <part of speech>
Definition: <clear, concise definition>
Examples:
1. <example sentence>
2. <example sentence>
Tip: <short mnemonic or memory aid>

Ensure the response follows this order exactly and uses these labels. Always provide exactly two example sentences.

For other vocabulary-related requests (e.g., definitions, comparisons, usage questions), respond clearly and helpfully in plain text without using the structured format.

Keep explanations simple, engaging, and practical. Avoid overly technical language.
Avoid very common words when suggesting new vocabulary; prefer slightly advanced vocabulary."""
    messages: List[Message] = [
        {"role": "system", "content": system_message}
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key not found. Check your .env file.")
        sys.exit(1)
    client = OpenAI(api_key = api_key)

    s1 = "She finished the assignment quickly."
    s2 = "She completed the task fast."
    #s3 = "I love programming in Python"
    e1 = get_embedding(client, s1)
    e2 = get_embedding(client, s2)
    #e3 = get_embedding(client, s3)

    sim_12 = cosine_similarity(e1, e2)
    #sim_13 = cosine_similarity(e1, e3)
    #sim_23 = cosine_similarity(e2, e3)

    print(sim_12)
    #print(sim_13)
    #print(sim_23)



    # while True:
    #     user_input = input("You: ")

    #     if user_input.lower() in ["exit", "quit"]:
    #         break

    #     # Create a *temporary* message list
    #     candidate_messages = messages + [
    #         {"role": "user", "content": user_input}
    #     ]

    #     success, response = get_openai_response(client, candidate_messages)

    #     if not success:
    #         print("Error:", response)
    #         continue  # nothing mutated → safe

    #     # Only now commit to history
    #     messages.append({"role": "user", "content": user_input})
    #     messages.append({"role": "assistant", "content": response})

    #     print("Assistant:", response)

if __name__ == "__main__":
    main()