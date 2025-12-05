from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate.from_messages([
    ('system','you are helpful {domain} expert'),
    ('human','explain in simple terms,what is {topic}')
])

prompt=chat_template.invoke({'domain':'cricket','topic':'dusara'})
print(prompt)