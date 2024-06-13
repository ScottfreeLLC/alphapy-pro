import openai
from openai import OpenAI

api_key = 'sk-proj-Z4GhEjvmru2IkkKpoH4RT3BlbkFJHbl3XwZSXep5kPE8mIZ0'
client = OpenAI(api_key=api_key)

# Set your OpenAI API key

# Define a function to test the API key
def test_openai_api_key(prompt):
    try:
        # Make a simple API request to the OpenAI API
        response = openai.chat.completions.create(model="gpt-4",
            messages=[
                    {"role": "system", "content": "Hello"},
                    {"role": "user", "content": prompt},
                ])
        # Print the response
        print("API Key is working. Response from OpenAI:")
        print(response.choices[0])
    except openai.AuthenticationError:
        print("Invalid API key. Please check your API key and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test function

prompt = "Write a Python moving average crossover system"
test_openai_api_key(prompt)
