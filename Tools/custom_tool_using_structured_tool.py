from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class Multiply_input(BaseModel):
    a:int=Field(required=True,description="the first number to add")
    b:int=Field(requied=True,description="the second number to add")

def multiply_func(a:int,b:int)->int:
    return a*b


multiply_tool=StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="multiply two numbers",
    args_schema=Multiply_input
)

result=multiply_tool.invoke({'a':3,'b':5})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
