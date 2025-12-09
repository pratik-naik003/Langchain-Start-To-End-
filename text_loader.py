from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()



loader=TextLoader('AI.txt')
docs=loader.load()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt=PromptTemplate(
    template='summarize the following poem {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

chain=prompt |model |parser
result=chain.invoke(docs[0].page_content)
print(result)





