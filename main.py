from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 

model = OllamaLLM(model="tinyllama")

template = """
You are an expert in economics and budget analysis.

Here is a database with 2020 budget for Mexico: {reviews}

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
    # Retrieve relevant documents for the question
    results = retriever.get_relevant_documents(question) 
    reviews = "\n\n".join([doc.page_content for doc in results])
    result = chain.invoke ({
    "reviews":reviews, 
    "question":question
    })
    print(result)

