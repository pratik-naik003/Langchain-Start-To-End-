from email import message
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage

load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

messages=[
    SystemMessage(content="you are a helpful AI assistent"),
    HumanMessage(content='tell me about langchainin two lines')
]
result=model.invoke(messages)
messages.append(result.content)
print(messages)







