from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence 
from langchain_core.runnables import RunnableParallel


load_dotenv()
prompt1=PromptTemplate(
    template='Generate a tweet about {text} ',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template='Generate linkdin post about text{text}',
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkdin_post': RunnableSequence(prompt2,model,parser)
})

result=parallel_chain.invoke({'text':'research intership in google company'})

print(result['tweet'])

print(result['linkdin_post'])


