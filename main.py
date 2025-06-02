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

result = chain.invoke ({
    "reviews":[], 
    "question":"what are the main insights for this budget?"
    })


print(result)

