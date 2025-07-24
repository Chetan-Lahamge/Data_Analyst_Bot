import os
from openai import OpenAI
from llm_confidence.logprobs_handler import LogprobsHandler
import dotenv
import sys

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Add current directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Initialize the LogprobsHandler
logprobs_handler = LogprobsHandler()

# Use API key from config file
api_key = config.OPENAI_API_KEY

# Set up your OpenAI client with your API key
client = OpenAI(api_key=api_key)

def get_completion(
        messages: list[dict[str, str]],
        model: str = "gpt-4o",
        max_tokens=500,
        temperature=0,
        stop=None,
        seed=42,
        response_format=None,
        logprobs=None,
        top_logprobs=None,
):
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if response_format:
        params["response_format"] = response_format

    completion = client.chat.completions.create(**params)
    return completion

def test_question(question):
    print(f"\n====== Testing Question: {question} ======")
    
    # For SQL-related questions, we'll format the output as JSON
    is_sql_related = any(keyword in question.lower() for keyword in ["sql", "database", "query", "table"])
    
    response_format = {'type': 'json_object'} if is_sql_related else None
    
    # Define a prompt for completion
    response_raw = get_completion(
        [{'role': 'user', 'content': question}],
        logprobs=True,
        response_format=response_format
    )

    # Print the output
    print("\nResponse content:")
    print(response_raw.choices[0].message.content)

    # Extract the log probabilities from the response
    has_logprobs = hasattr(response_raw.choices[0], 'logprobs')
    print(f"\nResponse has logprobs attribute: {has_logprobs}")
    
    if has_logprobs:
        response_logprobs = response_raw.choices[0].logprobs.content if response_raw.choices[0].logprobs else []
        
        print(f"Logprobs content length: {len(response_logprobs) if response_logprobs else 0}")
        
        if response_logprobs:
            print("\nLogprobs raw data structure type:", type(response_logprobs))
            
            # Print first few entries to see format
            if isinstance(response_logprobs, list):
                print("\nSample logprobs (first 3 entries):")
                for i, item in enumerate(response_logprobs[:3]):
                    print(f"Item {i}:", item)
                    if hasattr(item, '__dict__'):
                        print(f"Item {i} attributes:", item.__dict__)
            else:
                print("\nLogprobs data:", response_logprobs)
            
            # Check what the format_logprobs function produces
            print("\nFormatted logprobs result type:", type(logprobs_handler.format_logprobs(response_logprobs)))
            print("Formatted logprobs sample:", logprobs_handler.format_logprobs(response_logprobs)[:3])
            
            # Format the logprobs from OpenAI
            logprobs_formatted = logprobs_handler.format_logprobs(response_logprobs)
            
            # Process the log probabilities to get confidence scores
            confidence = logprobs_handler.process_logprobs(logprobs_formatted)
            
            # Print the confidence scores
            print("\nConfidence scores:")
            print(confidence)
            
            # Calculate average confidence
            if isinstance(confidence, dict) and confidence:
                avg_confidence = sum(confidence.values()) / len(confidence)
                print(f"\nAverage confidence: {avg_confidence:.4f}")
                print(f"Normalized to percentage: {avg_confidence * 100:.2f}%")
            elif isinstance(confidence, (int, float)):
                print(f"\nSingle confidence score: {confidence:.4f}")
                print(f"Normalized to percentage: {confidence * 100:.2f}%")
            else:
                print("\nNo confidence scores available")
        else:
            print("\nNo logprobs content available in the response")
    else:
        print("\nNo logprobs attribute in the response")

# List of sample questions to test
sample_questions = [
    "What is the capital of France?",
    "Write a SQL query to get all customers who made purchases in the last month.",
    "What is 15 multiplied by 27?",
    "Explain the concept of quantum computing in simple terms.",
    "What is the difference between a inner join and left join in SQL?"
]

# Test each question
if __name__ == "__main__":
    print("Confidence Score Demo")
    print("=====================")
    
    while True:
        print("\nOptions:")
        print("1. Run sample questions")
        print("2. Enter your own question")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            for question in sample_questions:
                test_question(question)
                input("\nPress Enter to continue to the next question...")
        elif choice == "2":
            user_question = input("\nEnter your question: ")
            test_question(user_question)
        elif choice == "3":
            print("\nExiting the demo. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.") 
