import spacy

nlp = spacy.load('en_core_web_sm')

from transformers import pipeline

# Loading ai responses instead of pre built
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

def chatbot_response(user_input):
    response = chatbot(user_input)
    return response[0]['generated_text']

# Looping the chat
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot_response(user_input)
    print(f"Bot: {response}")