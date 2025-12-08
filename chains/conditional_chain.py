from langchain_core.runnables import RunnableLambda, RunnableBranch
from html import parser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')



# Branch 1 → Weather Chain
def weather_response(input):
    return model.invoke(f"What is the weather like in {input['text']}?")

# Branch 2 → Normal Chat Chain
def general_response(input):
    return model.invoke(f"Answer this normally: {input['text']}")

# Condition function
def is_weather_query(input):
    return "weather" in input["text"].lower()

# Create a conditional chain
conditional_chain = RunnableBranch(
    (lambda i: is_weather_query(i), RunnableLambda(weather_response)),
    RunnableLambda(general_response)  # default branch
)

result = conditional_chain.invoke({"text": "to day weather of sangli"})
print(result.content)
