import requests
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sympy as sp
import re
import json
import os

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load DialoGPT model and tokenizer
print("Loading chatbot model...")
try:
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Chatbot model loaded.")
except Exception as e:
    print(f"Failed to load chatbot model: {e}")
    model, tokenizer, chatbot = None, None, None

# Memory storage
memory_file = 'memory.json'
response_log_file = 'response.json'

def load_memory():
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as file:
            try:
                content = file.read().strip()
                if content:
                    return json.loads(content)
            except json.JSONDecodeError:
                return {}
    return {}

def save_memory():
    with open(memory_file, 'w') as file:
        json.dump(memory, file)

def log_response(user_input, bot_response):
    log_entry = {"user_input": user_input, "bot_response": bot_response}
    if os.path.exists(response_log_file):
        with open(response_log_file, 'r+') as file:
            try:
                data = json.load(file)
                data.append(log_entry)
                file.seek(0)
                json.dump(data, file)
            except json.JSONDecodeError:
                file.seek(0)
                json.dump([log_entry], file)
    else:
        with open(response_log_file, 'w') as file:
            json.dump([log_entry], file)

memory = load_memory()

def solve_math_expression(expression):
    try:
        # Use sympy to parse and solve the math expression
        result = sp.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error in math computation: {e}"

def update_memory(key, value):
    # Save key-value pairs to memory
    memory[key] = value
    save_memory()

def retrieve_memory(key):
    # Retrieve information from memory
    return memory.get(key, None)

def replace_words_with_symbols(expression):
    expression = expression.lower()
    expression = expression.replace("plus", "+")
    expression = expression.replace("minus", "-")
    expression = expression.replace("times", "*")
    expression = expression.replace("divided by", "/")
    return expression

def extract_math_expression(user_input):
    # Replace words with symbols
    user_input = replace_words_with_symbols(user_input)
    # Use regular expression to find math expressions
    match = re.search(r'(\d+(\s*[\+\-\*\/]\s*\d+)+)', user_input)
    if match:
        return match.group(0)
    return None

def get_search_results(query):
    """Get search results from Google Custom Search API."""
    api_key = 'AIzaSyDg3raBDHsYYgzUt96U40z-x5EL502CTLs'
    search_engine_id = 'c0c21f9a67e4e4474'
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}'
    response = requests.get(url)
    data = response.json()
    
    # Check if search results are available
    if 'items' in data:
        results = []
        for item in data['items'][:3]:  # Get the top 3 results
            title = item.get('title')
            snippet = item.get('snippet')
            link = item.get('link')
            results.append(f"{title}\n{snippet}\n{link}\n")
        return "\n\n".join(results)
    return "Sorry, I couldn't find an answer to that."

def chatbot_response(user_input):
    # Extract and solve math expression if present
    math_expression = extract_math_expression(user_input)
    if math_expression:
        return solve_math_expression(math_expression)

    if chatbot is None:
        return "Chatbot model is not available."
    try:
        # Check if the user is providing their name
        if "my name is" in user_input.lower():
            name = user_input.split("my name is")[-1].strip()
            update_memory("name", name)
            response = f"Nice to meet you, {name}!"
            log_response(user_input, response)
            return response

        # Check for specific information requests
        if "my name" in user_input.lower():
            name = retrieve_memory("name")
            if name:
                response = f"Your name is {name}."
            else:
                response = "I don't know your name yet. What is your name?"
            log_response(user_input, response)
            return response

        # Check for other types of information
        if "remember" in user_input.lower():
            parts = user_input.split("remember")
            if len(parts) > 1:
                info = parts[1].strip()
                key_value = info.split("is")
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    update_memory(key, value)
                    response = f"Got it! I'll remember that {key} is {value}."
                    log_response(user_input, response)
                    return response

        # Check if the user is asking for remembered information
        if "what is" in user_input.lower():
            key = user_input.split("what is")[-1].strip()
            value = retrieve_memory(key)
            if value:
                response = f"{key} is {value}."
            else:
                response = f"I don't know what {key} is yet."
            log_response(user_input, response)
            return response

        # Check if the user is asking a factual question
        search_phrases = ["search for", "find", "look up", "who is", "what is", "tell me about"]
        if any(phrase in user_input.lower() for phrase in search_phrases):
            query = re.sub('|'.join(search_phrases), '', user_input, flags=re.IGNORECASE).strip()
            response = get_search_results(query)
            log_response(user_input, response)
            return response

        # Encoding the user input with attention_mask
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Create attention mask (1 for actual tokens, 0 for padding tokens)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create a tensor of ones
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # Set padding tokens to 0

        # Generating a response with the chatbot
        chat_history_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the response
        chatbot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Log the response
        log_response(user_input, chatbot_response)
        
        return chatbot_response
    except Exception as e:
        response = "Sorry, I couldn't process that."
        log_response(user_input, response)
        return response

# Looping the chat
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye! Have a great day!")
        break
    elif not user_input.strip():
        print("Bot: Please enter something.")
        continue
    response = chatbot_response(user_input)
    print(f"Bot: {response}")
