from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model="qwen3:0.6b")

template = """
You are an expert in economics and budget analysis.

Here is a database with 2018 budget for Mexico: {reviews}

Here is the question to answer: {question}

Reply in English.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model 

while True: 
    print("\n\n-------------------------------------------------------------------------------")
    question = input("ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    
    result = chain.invoke ({
    "reviews":[], 
    "question":question"
    })
    print(result)

