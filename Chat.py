import spacy

nlp = spacy.load('en_core_web_sm')

from transformers import pipeline

# Load pre-trained conversational model
print("Loading chatbot model...")
try:
    chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
    print("Chatbot model loaded.")
except Exception as e:
    print(f"Failed to load chatbot model: {e}")
    chatbot = None

def chatbot_response(user_input):
    print(f"User input: {user_input}")
    if chatbot is None:
        return "Chatbot model is not available."
    try:
        response = chatbot(user_input)
        print(f"Chatbot response: {response}")
        return response[0]['generated_text']
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