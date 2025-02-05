import os
import requests
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sympy as sp
import re
import json
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import ttk, scrolledtext
import sqlite3
from collections import deque

response_log_file = "response_log.json"

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

# Initialize SQLite database
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

def update_memory(key, value):
    print(f"Updating memory: {key} = {value}")
    cursor.execute('''INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)''', (key, value))
    conn.commit()

def retrieve_memory(key):
    print(f"Retrieving memory for key: {key}")
    cursor.execute('''SELECT value FROM memory WHERE key = ?''', (key,))
    result = cursor.fetchone()
    print(f"Retrieved value: {result[0] if result else None}")
    return result[0] if result else None

def log_response(user_input, bot_response):
    log_entry = {"user_input": user_input, "bot_response": bot_response}
    if os.path.exists(response_log_file):
        try:
            with open(response_log_file, 'r+', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
                data.append(log_entry)
                file.seek(0)
                json.dump(data, file, ensure_ascii=False, indent=4)
        except UnicodeDecodeError:
            with open(response_log_file, 'w', encoding='utf-8') as file:
                json.dump([log_entry], file, ensure_ascii=False, indent=4)
    else:
        with open(response_log_file, 'w', encoding='utf-8') as file:
            json.dump([log_entry], file, ensure_ascii=False, indent=4)

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
    api_key = os.getenv('GOOGLE_API_KEY')
    search_engine_id = os.getenv('SEARCH_ENGINE_ID')
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

# Initialize chat history
chat_history = deque(maxlen=10)

def chatbot_response(user_input):
    print(f"Processing input: {user_input}")
    
    if chatbot is None:
        return "Chatbot model is not available."
        
    try:
        # Handle math expressions
        math_expression = extract_math_expression(user_input)
        if math_expression:
            return solve_math_expression(math_expression)

        # Handle translation
        if "translate" in user_input.lower():
            parts = user_input.split("translate")
            if len(parts) > 1:
                text_and_language = parts[1].strip().split("to")
                if len(text_and_language) == 2:
                    text = text_and_language[0].strip()
                    target_language = text_and_language[1].strip()
                    return translate_text(text, target_language)

        # Handle search queries
        search_phrases = ["search for", "find", "look up", "who is", "tell me about"]
        if any(phrase in user_input.lower() for phrase in search_phrases):
            query = parse_query_with_spacy(user_input)
            return get_search_results(query)

        # Handle memory storage
        if "my favorite color is" in user_input.lower():
            color = user_input.split("my favorite color is")[-1].strip()
            update_memory("favorite_color", color)
            return f"Got it! Your favorite color is {color}."

        # Handle memory retrieval
        if "remember" in user_input.lower():
            parts = user_input.split("remember")
            if len(parts) > 1:
                key = parts[1].strip()
                value = retrieve_memory(key)
                return f"Yes, {key} is {value}." if value else f"I don't remember {key}."

        # Handle general conversation with DialoGPT
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        chat_output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(chat_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm not sure how to respond to that."

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

chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12), bg="#23272a", fg="white")
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=10, fill=tk.X, expand=True)

user_entry = ttk.Entry(input_frame, font=("Arial", 12))
user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
user_entry.bind("<Return>", send_message)

send_button = ttk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()