import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

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

def chatbot_response(user_input):
    if chatbot is None:
        return "Chatbot model is not available."
    try:
        # Encoding the user input with attention_mask
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Create attention mask (1 for actual tokens, 0 for padding tokens)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create a tensor of ones
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # Set padding tokens to 0

        # Generating a response with the chatbot
        chat_history_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the response
        chatbot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
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
