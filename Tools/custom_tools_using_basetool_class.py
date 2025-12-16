from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel,Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

class Multiplytool(BaseTool):
    name:str="multiply"
    description:str="multiply two numbers"
    
    args_schema:Type[BaseModel]=MultiplyInput
    
    def _run(self,a:int,b:int)->int:
        return a*b

mutiply_tool=Multiplytool()

result=mutiply_tool.invoke({'a':3,'b':3})
print(result)
print(mutiply_tool.description)
print(mutiply_tool.args)

