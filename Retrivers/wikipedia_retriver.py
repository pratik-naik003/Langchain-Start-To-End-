from langchain_community.retrievers import WikipediaRetriever

#intializa the retriever
retriever=WikipediaRetriever(top_k_results=3,lang='en')

#query
query="what is langchain"
docs=retriever.invoke(query)
for i in docs:
    print(i.page_content)