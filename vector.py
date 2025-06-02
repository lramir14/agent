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
add_documents = len(existing_docs['documents']) == 0

print(f"🧠 Existing documents: {len(existing_docs['documents'])}")


if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        doc_text = f"{row['CICLO']} {row['DESC_PARTIDA_ESPECIFICA']} {row['MONTO_APROBADO']}"
        document = Document(page_content=doc_text, metadata={"date": row['CICLO']}, id=str(i))
        documents.append(document)
        ids.append(str(i))
        if i < 3:
            print(f"📝 Preview doc {i}: {doc_text[:80]}")

    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()
    print(f"✅ Added and persisted {len(documents)} documents.")
else:
    print("📦 Vector store already exists, skipping add.")
# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Test retrieval
test_query = "educación 2020 monto pagado"
results = retriever.get_relevant_documents(test_query)
print(f"🔍 Retrieved {len(results)} documents:")
for doc in results[:3]:
    print(doc.page_content[:200], "...\n")
