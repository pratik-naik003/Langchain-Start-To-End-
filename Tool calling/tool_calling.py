from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

#tool create 
@tool
def multiply(a:int,b:int)->int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b
# print(multiply.invoke({'a':3,'b':4}))

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

#tool binding

model_with_tool=model.bind_tools([multiply])

#tool calling


query=HumanMessage("can you multiply 4000 with 4")

messages=[query]

result=model_with_tool.invoke(messages)

messages.append(result)


tool_result=multiply.invoke(result.tool_calls[0])


messages.append(tool_result)

print(model_with_tool.invoke(messages).content)










