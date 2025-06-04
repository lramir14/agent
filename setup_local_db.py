from pathlib import Path
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import lancedb
import numpy as np
import warnings

# Completely disable OpenAI-related warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*openai.*")

# ─── (1) Define the Ollama‐based embedding function ────────────────────────────
class LocalEmbeddingFunction:
    def __init__(self, model_name="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.ndims = 768
        
    def __call__(self, texts):
        if isinstance(texts, str):
            return self.embeddings.embed_query(texts)
        return self.embeddings.embed_documents(texts)
        
    def generate_embeddings(self, texts):
        """Bypass any OpenAI-related code paths"""
        if isinstance(texts, str):
            return np.array([self.embeddings.embed_query(texts)])
        return np.array(self.embeddings.embed_documents(texts))

embedding_func = LocalEmbeddingFunction()


# ─── (2) LanceDB schema ───────────────────────────────────────────────────────
class Document(LanceModel):
    id: str
    text: str = embedding_func.SourceField()
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()


# ─── (3) Text‐chunking helper ───────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# ─── (4) Create (or overwrite) the LanceDB table ───────────────────────────────
def create_lancedb_table(db_path: str, table_name: str, overwrite: bool = True):
    db = lancedb.connect(db_path)
    
    # Create table with explicit schema
    schema = {
        "id": str,
        "text": str,
        "vector": lancedb.vector(embedding_func.ndims)
    }
    
    table = db.create_table(
        table_name,
        schema=schema,
        mode="overwrite" if overwrite else "create"
    )
    
    # Manually inject our embedding function
    table._embedding_function = embedding_func
    return table


# ─── (5) Load .md files into the table ──────────────────────────────────────────
def add_documents_to_table(table: LanceTable, knowledge_base_dir: str):
    docs = []
    knowledge_base = Path(knowledge_base_dir)

    for md_file in knowledge_base.glob("*.md"):
        print(f"Processing {md_file.name}")
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{md_file.stem}_{i}"
                docs.append({"id": doc_id, "text": chunk})

    if docs:
        table.add(docs)
        print(f"Added {len(docs)} documents to the table.")
    else:
        print("No documents found.")


# ─── (6) Load CSV rows into the table ──────────────────────────────────────────
def add_csv_to_table(table: LanceTable, csv_path: str):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    docs = []
    for idx, row in df.iterrows():
        row_dict = row.dropna().to_dict()
        text = "\n".join(f"{key}: {value}" for key, value in row_dict.items())
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{Path(csv_path).stem}_{idx}_{i}"
            docs.append({"id": doc_id, "text": chunk})

    if docs:
        table.add(docs)
        print(f"Added {len(docs)} CSV documents.")
    else:
        print("No CSV data added.")


# ─── (7) Search & rerank ───────────────────────────────────────────────────────
def retrieve_similar_docs(
    table: LanceTable,
    query: str,
    query_type: str = "hybrid",
    limit: int = 100,
    reranker_weight: float = 0.7
):
    """
    ⚠️ Important: There must be NO `embedding=…` argument here.
    `table.embedding_function` was already set in setup_lancedb(), so
    `table.search(query, query_type=…)` will call embedding_func internally.
    """
    reranker = LinearCombinationReranker(weight=reranker_weight)

    # Correct: do not pass embedding=embedding_func
    results = (
        table
        .search(query, query_type=query_type)
        .rerank(reranker=reranker)
        .limit(limit)
        .to_list()
    )
    return results


# ─── (8) Entire pipeline setup ───────────────────────────────────────────────────
def setup_lancedb():
    db_path = "./db"
    table_name = "knowledge"
    knowledge_base_dir = "./knowledge-base"
    csv_path = "./data/presupuesto_mexico__2020.csv"

    table = create_lancedb_table(db_path, table_name, overwrite=True)

    # ⚠️ Assign the embedding function exactly once here:
    table.embedding_function = embedding_func

    # Add markdown files and CSV rows
    add_documents_to_table(table, knowledge_base_dir)
    add_csv_to_table(table, csv_path)

    return table


#if __name__ == "__main__":
 #   table = setup_lancedb()
  #  results = retrieve_similar_docs(table, "education")
   # for doc in results[:3]:
     #   print(f"ID: {doc['id']}\nText: {doc['text'][:200]}...\n")
