from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os 
import pandas as pd 


df = pd.read_csv("data/presupuesto_mexico__2020.csv", low_memory=False)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain.db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content = str(row["CICLO"]) + " " + str(row["DESC_PARTIDA_ESPECIFICA"]) + " " + \
               str(row["DESC_PARTIDA_GENERICA"]) + " " + str(row["DESC_TIPOGASTO"]) + " " + \
               str(row["ID_ENTIDAD_FEDERATIVA"]) + " " + str(row["MONTO_APROBADO"]) + " " + \
               str(row["MONTO_MODIFICADO"]) + " " + str(row["MONTO_PAGADO"]),
            metadata={"date":row["CICLO"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document) 
        
        
vector_store = Chroma(
    collection_name="budget",
    persist_directory=db_location,
    embedding_function=embeddings
)

existing_docs = vector_store.get()
print(f"🧠 Vector DB contains {len(existing_docs['documents'])} documents.")
    

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()
    print(f"✅ Added {len(documents)} documents to the vector database.")
else:
    print(f"📂 Loaded existing vector database from {db_location}.")
    
    
#retrieve to search and lookup for relevant documents ## specify how many key articles or results to retrieve back

retriever = vector_store.as_retriever(
    search_kwargs={"k":10}
)

# Test retrieval
results = retriever.get_relevant_documents("educación 2020 monto pagado")
print(f"🔍 Retrieved {len(results)} documents:")
for doc in results[:3]:
    print(doc.page_content[:200], "...\n")