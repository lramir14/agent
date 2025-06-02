from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os 
import pandas as pd 


df = pd.read_csv("data/presupuesto_mexico__2020.csv", low_memory=False)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db_location = "./chroma_langchain.db"

vector_store = Chroma(
    collection_name="budget",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Check if the vector store already has documents
existing_docs = vector_store.get()
print(f"🧠 Vector DB contains {len(existing_docs['documents'])} documents.")

# Only add documents if DB is empty
if len(existing_docs['documents']) == 0:
    print("Adding documents to vector store...")

    documents = []
    ids = []

    for i, row in df.iterrows():
        content = (
            f"{row['CICLO']} {row['DESC_PARTIDA_ESPECIFICA']} {row['DESC_PARTIDA_GENERICA']} "
            f"{row['DESC_TIPOGASTO']} {row['ID_ENTIDAD_FEDERATIVA']} {row['MONTO_APROBADO']} "
            f"{row['MONTO_MODIFICADO']} {row['MONTO_PAGADO']}"
        )
        document = Document(
            page_content=str(content),
            metadata={"date": row["CICLO"]},
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()
    print(f"✅ Added {len(documents)} documents to the vector database.")
else:
    print(f"📂 Loaded existing vector database from {db_location}.")
# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Test retrieval
test_query = "educación 2020 monto pagado"
results = retriever.get_relevant_documents(test_query)
print(f"🔍 Retrieved {len(results)} documents:")
for doc in results[:3]:
    print(doc.page_content[:200], "...\n")
