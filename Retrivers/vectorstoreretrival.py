from webbrowser import Chrome
from langchain_community.vectorstores import Chroma
from importlib import metadata
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from openai import embeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#your source model 
documents=[
    Document(page_content="langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

#create chroma vector store in memory
vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="new_my_collection"
)

#convert vectorstore into a rettriever
retriever=vector_store.as_retriever(search_kwargs={"k":2})

#query
query="what is chrome used for?"
results=retriever.invoke(query)
for i in results:
    print(i.page_content)
    
