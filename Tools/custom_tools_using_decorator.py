from langchain_core.tools import tool

#step 1 -create a function
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

#Step 2 add two hints

def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b

#step - add tool decorator 
@tool
def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b

result=multiply.invoke({"a":3,"b":5})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
