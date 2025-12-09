from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader



load_dotenv()

# loader=PyPDFLoader('deep_learning_simple.pdf')
loader=PyPDFLoader('Pratik Naik Resume .pdf-1.pdf.pdf')
docs=loader.load()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt=PromptTemplate(
    template='suggest me job role for this resume {resume}',
    input_variables=['resume']
)

parser=StrOutputParser()

chain=prompt |model|parser

result=chain.invoke(docs[0].page_content)
print(result)
