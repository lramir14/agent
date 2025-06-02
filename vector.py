from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os 
import pandas as pd 


df = pd.read_csv("data/presupuesto_mexico__2020.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain.db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content= row["CICLO"] + row["DESC_PARTIDA_ESPECIFICA"] + row["DESC_PARTIDA_GENERICA"] +
            row["DESC_TIPOGASTO"] + row["ID_ENTIDAD_FEDERATIVA"] + row["MONTO_APROBADO"] + row["MONTO_MODIFICADO"] + 
            row["MONTO_PAGADO"],
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

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

#retrieve to search and lookup for relevant documents ## specify how many key articles or results to retrieve back

retriever = vector_store.as_retriever(
    search_kwargs={"k":10}
)