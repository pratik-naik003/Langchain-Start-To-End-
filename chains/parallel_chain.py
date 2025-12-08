from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os


load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# llm=HuggingFaceEndpoint(
#          repo_id="moonshotai/Kimi-K2-Instruct-0905",
#         huggingfacehub_api_token=api_key)

model1=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

model2=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# model3=ChatHuggingFace(llm=llm,temprature=0.1,max_completion_tokens=5)


prompt1=PromptTemplate(
    template='generate short and simple notes from the following text  {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='generate 5 short question from the following text  {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='merge the provided notes and quiz into single also and add multiple choice questions  and at filnal give the answers of the all the questions document \nnotes-> {notes}/n quize->{quiz}',
    input_variables=['notes','quiz']
)


parser=StrOutputParser()

parall_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

# merged_chain=parall_chain|model1|parser


# chain=parall_chain |merged_chain
merged_chain = parall_chain | prompt3 | model1 | parser
text="""Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It fits a straight line that best represents the data. The line predicts outcomes by minimizing errors, helping understand trends, correlations, and future values from existing patterns with insights."""

result=merged_chain.invoke({'text':text})
print(result)




