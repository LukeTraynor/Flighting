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

def load_memory():
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as file:
            return json.load(file)
    return {}

def save_memory():
    with open(memory_file, 'w') as file:
        json.dump(memory, file)

memory = load_memory()

def solve_math_expression(expression):
    try:
        # Use sympy to parse and solve the math expression
        result = sp.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error in math computation: {e}"

def update_memory(user_input, bot_response):
    # Save user inputs and bot responses to memory
    memory[user_input] = bot_response
    save_memory()

def retrieve_memory(user_input):
    # Retrieve information from memory
    return memory.get(user_input, None)

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

def chatbot_response(user_input):
    # Extract and solve math expression if present
    math_expression = extract_math_expression(user_input)
    if math_expression:
        return solve_math_expression(math_expression)

    if chatbot is None:
        return "Chatbot model is not available."
    try:
        # Check if input already exists in memory
        remembered_answer = retrieve_memory(user_input)
        if remembered_answer:
            return f"I remember you asked that before: {remembered_answer}"

        # Encoding the user input with attention_mask
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Create attention mask (1 for actual tokens, 0 for padding tokens)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create a tensor of ones
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # Set padding tokens to 0

        # Generating a response with the chatbot
        chat_history_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the response
        chatbot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Store the response in memory
        update_memory(user_input, chatbot_response)
        
        return chatbot_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't process that."

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