from dotenv import load_dotenv
import os
from openai_client import get_openai_response

load_dotenv()

def main():
    user_input = input("Enter your messaage:")
    response = get_openai_response(user_input)
    print(response)

if __name__ == "__main__":
    main()