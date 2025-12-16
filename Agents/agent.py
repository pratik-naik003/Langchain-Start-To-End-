from langchain_core.tools import tool
import requests
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent
# OR for older reasoning styles
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

load_dotenv()

search_tool = DuckDuckGoSearchRun()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=3fac7472aa5828a1f8c3f8ada3f26d6d&query={city}'

  response = requests.get(url)

  return response.json()

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=model,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)
# Step 5: Invoke
response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)