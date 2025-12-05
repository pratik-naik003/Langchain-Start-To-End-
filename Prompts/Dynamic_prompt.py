from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header('Research Assistent')

paper_input=st.selectbox("select areaserch paper name",["attention is all you need","BERT : pre traning of deep bidirectional transformers"])

style_input=st.selectbox("select explanation style",["with simple funny jokes","tecnical","funny way"])

length_input=st.selectbox("select explanation length",["5 lines","10 lines","50 lines"])


template=PromptTemplate(
    template="""Please summarize the research paper, titled {paper_input} with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}
Mathematical Details:
Include relevant mathematical equations if present in the paper.
Explain the mathematical concepts using simple, intuitive code snippets where applicable.
Analogies:
Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length. 
""",
input_variables=['paper_input','style_input','length_input'])
user_inut=st.text_input('Enter your prompt')


#fill the placeholders
promt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    
})
if st.button('Summarize'):
    result=model.invoke(promt)
    st.write(result.content)


