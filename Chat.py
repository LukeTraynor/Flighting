import requests
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sympy as sp
import re
import json
import os
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import ttk, scrolledtext
import sqlite3

response_log_file = "chatbot_memory.db"
chat_log_file = "chat_log.json"

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

# Initializing SQLite database
conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()

# Create memory table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory (
        key TEXT PRIMARY KEY,
        value TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Function to store or update memory
def update_memory(key, value):
    cursor.execute('INSERT INTO memory (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?;', (key, value, value))
    conn.commit()

# Function to retrieve memory
def retrieve_memory(key):
    cursor.execute('SELECT value FROM memory WHERE key = ?', (key,))
    result = cursor.fetchone()
    return result[0] if result else None

# Function to retrieve all stored memories
def retrieve_all_memories():
    cursor.execute('SELECT key, value FROM memory')
    rows = cursor.fetchall()
    if rows:
        return ", ".join([f"{key}: {value}" for key, value in rows])
    return "I don't remember anything yet."

# Function to append data instead of overwriting
def append_memory(key, new_info):
    existing_value = retrieve_memory(key)
    if existing_value:
        new_value = existing_value + " " + new_info
    else:
        new_value = new_info
    update_memory(key, new_value)

# Function to forget a specific memory
def forget_memory(key):
    cursor.execute('DELETE FROM memory WHERE key = ?', (key,))
    conn.commit()

# Function to delete all memories
def forget_all_memories():
    cursor.execute('DELETE FROM memory')
    conn.commit()


def log_response(user_input, bot_response):
    log_entry = {"user_input": user_input, "bot_response": bot_response}
    
    if os.path.exists(chat_log_file):
        try:
            with open(chat_log_file, 'r+', encoding='utf-8') as file:
                data = json.load(file)  # Load existing log
                data.append(log_entry)  # Append new entry
                file.seek(0)  # Move cursor to beginning
                json.dump(data, file, indent=4)  # Write updated log
        except (json.JSONDecodeError, UnicodeDecodeError):  
            with open(chat_log_file, 'w', encoding='utf-8') as file:
                json.dump([log_entry], file, indent=4)
    else:
        with open(chat_log_file, 'w', encoding='utf-8') as file:
            json.dump([log_entry], file, indent=4)

def solve_math_expression(expression):
    try:
        # Use sympy to parse and solve the math expression
        result = sp.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error in math computation: {e}"

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

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")



def get_search_results(query):
    """Get search results from Google Custom Search API and summarize."""
    api_key = ('AIzaSyDg3raBDHsYYgzUt96U40z-x5EL502CTLs')
    search_engine_id = ('c0c21f9a67e4e4474')
    if not api_key or not search_engine_id:
        return "API key or search engine ID is not set."
    
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        
        # Check if search results are available
        if 'items' in data:
            snippets = []
            for item in data['items'][:5]:  # Get the top 5 results
                snippet = item.get('snippet', '')
                if snippet:
                    snippets.append(snippet)
            
            # Combine snippets into one text
            combined_text = " ".join(snippets)
            
            # Summarize the combined text
            if combined_text:
                summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
                return summary[0]['summary_text']
        
        return "Sorry, I couldn't find a summary for that."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while searching: {e}"

def parse_query_with_spacy(user_input):
    """Parse the user input with spaCy to extract the search query."""
    doc = nlp(user_input)
    query = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            query.append(token.text)
    return ' '.join(query)

def translate_text(text, target_language):
    """Translate text to the target language using deep_translator."""
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        return f"An error occurred while translating: {e}"

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
        if "my name" in user_input.lower() and "is" not in user_input.lower():
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
        if "what is" in user_input.lower() and "search for" not in user_input.lower():
            key = user_input.split("what is")[-1].strip()
            value = retrieve_memory(key)
            if value:
                response = f"{key} is {value}."
            else:
                response = f"I don't know what {key} is yet."
            log_response(user_input, response)
            return response

        # Check if the user is asking a factual question
        search_phrases = ["search for", "find", "look up", "who is", "tell me about"]
        if any(phrase in user_input.lower() for phrase in search_phrases):
            query = parse_query_with_spacy(user_input)
            response = get_search_results(query)
            log_response(user_input, response)
            return response

        # Check if the user is asking for translation
        if "translate" in user_input.lower():
            parts = user_input.split("translate")
            if len(parts) > 1:
                text_and_language = parts[1].strip().split("to")
                if len(text_and_language) == 2:
                    text = text_and_language[0].strip()
                    target_language = text_and_language[1].strip()
                    response = translate_text(text, target_language)
                    log_response(user_input, response)
                    return response

        # Generate a response using the DialoGPT model
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Create attention mask (1 for actual tokens, 0 for padding tokens)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # Set padding tokens to 0

        # Generate a response with the chatbot
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


def send_message(event=None):
    user_input = user_entry.get()
    if user_input.lower() == 'exit':
        root.quit()
    elif not user_input.strip():
        chat_display.config(state=tk.NORMAL)  # Enable text widget
        chat_display.insert(tk.END, "Bot: Please enter something.\n")
        chat_display.config(state=tk.DISABLED)  # Disable text widget
    else:
        chat_display.config(state=tk.NORMAL)  # Enable text widget
        chat_display.insert(tk.END, f"You: {user_input}\n", "user_message")
        response = chatbot_response(user_input)
        chat_display.insert(tk.END, f"Bot: {response}\n")
        chat_display.config(state=tk.DISABLED)  # Disable text widget
        chat_display.see(tk.END)  # Auto-scroll to the latest message
        user_entry.delete(0, tk.END)



root = tk.Tk()
root.title("Chatbot")
root.geometry("500x600")
root.configure(bg="#2c2f33")


style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TEntry", padding=5)
style.configure("TFrame", background="#2c2f33")


chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12), bg="#23272a", fg="white")
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_display.tag_config("user_message", foreground="#00ffcc")

input_frame = ttk.Frame(root)
input_frame.pack(pady=10, fill=tk.X, padx=10)

user_entry = ttk.Entry(input_frame, font=("Arial", 12))
user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
user_entry.bind("<Return>", send_message)
send_button = ttk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()