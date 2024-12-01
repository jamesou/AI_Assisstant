import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
# Load environment variables from .env file
load_dotenv()
# This is the default and can be omitted
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True
 

if check_openai_api_key(OPENAI_API_KEY):
    print("Valid OpenAI API key.")
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ]
    )

    print(completion.choices[0].message)
else:
    print("Invalid OpenAI API key.")