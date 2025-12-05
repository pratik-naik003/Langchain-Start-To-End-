from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header('Research Assistent')

user_inut=st.text_input('Enter your prompt')

if st.button('Summarize'):
    result=model.invoke(user_inut)
    st.write(result.content)




