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


doc1=Document(page_content="virat kohli is an indian cricketer and a graeat batsman.",metadata={"team":"royal challenges banglore"})

doc2=Document(page_content="rohit sharma is the capital of mumbai indians.",metadata={"team":"Mumbai indians"})

doc3=Document(page_content="MS dhoni is a former captain of chennai super king.",metadata={"team":"chennai super king"})

doc4=Document(page_content="jasprit bumrah is a bowler for mumbai indians",metadata={"team":"mumbai indians"})

doc5 = Document(page_content="Ravindra Jadeja is an all-rounder in Chennai Super Kings.", metadata={"team": "Chennai Super Kings"})

doc=[doc1,doc2,doc3,doc4,doc5]

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store=Chroma(embedding_function=embeddings,persist_directory="my_chromadb",collection_name="sample")

# Add Documents
vector_store.add_documents(doc)

# View Data
result=vector_store.get(include=["embeddings", "documents", "metadatas"])

#Similarity Search
query = "Who among these is former captain?"
result2=vector_store.similarity_search(query, k=2)

#Similarity Search with Score
result3=vector_store.similarity_search_with_score(query, k=2)

#Filter by Metadata

result4=vector_store.similarity_search(query="player", k=10, filter={"team": "Chennai Super Kings"})

#Update a Document
updated_doc = Document(page_content="Virat Kohli, former captain of RCB, is known for his aggressive leadership.", metadata={"team": "Royal Challengers Bangalore"})

result5=vector_store.update_document(document_id="1", document=updated_doc)

#Delete a Document
vector_store.delete(ids=["1"])
print(result)
