import spacy

nlp = spacy.load('en_core_web_sm')

def chatbot_response(user_input):
    doc = nlp(user_input)
    
    # Basic Response
    if 'hello' in user_input.lower() or 'hi' in user_input.lower():
        return "Hello! How can I assist you today?"
    elif 'how are you' in user_input.lower():
        return "I'm just a bot, but I'm doing well. How about you?"
    elif 'bye' in user_input.lower():
        return "Goodbye! Have a great day!"
    else:
        return "Sorry, I didn't understand that. Could you rephrase?"

# Looping the chat
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot_response(user_input)
    print(f"Bot: {response}")