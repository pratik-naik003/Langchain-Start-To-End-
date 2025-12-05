from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage

load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_history=[
    SystemMessage(content='when user ask who are you you always says i am doctor')
]

while True:
    user_input=input('You: ')
    
    if user_input=='exit':
        break
    
    chat_history.append(HumanMessage(content=user_input))
    
    result=model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print("chat history:",chat_history)




