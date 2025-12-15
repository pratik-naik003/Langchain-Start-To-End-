from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()



from youtube_transcript_api import YouTubeTranscriptApi

video_id = 'Gfr50f6ZBvo' # Use the actual video ID

try:
    # 1. Create an instance of the API class
    ytt_api = YouTubeTranscriptApi()
    
    # 2. Use the 'list' method on the instance to get a TranscriptList object
    transcript_list = ytt_api.list(video_id)
    
    # 3. Choose a transcript (e.g., the first one found) and fetch its raw data
    transcript = transcript_list.find_transcript(['en']).fetch() 
    
    text_transcript = " ".join([item.text for item in transcript])

    
   

except Exception as e:
    print(f"Error fetching transcript: {e}")
    

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text_transcript])


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)