from langchain_community.document_loaders import WebBaseLoader
from pydantic_core import Url
url='https://en.wikipedia.org/wiki/Walchand_College_of_Engineering,_Sangli'
loader=WebBaseLoader(url)
doc=loader.load()



print(doc[0].page_content)