from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel



load_dotenv()
def word_count(text):
    return len(text.split())

prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser=StrOutputParser()

joke_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnableSequence(prompt,model,parser),
    'word_count':RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_chain,parallel_chain)

result=final_chain.invoke({'topic':'AI'})
print(result)


