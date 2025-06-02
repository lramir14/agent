from pathlib import Path
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from lancedb.table import Table as LanceTable

# Create a custom embedding function wrapper
class OllamaEmbeddingFunction:
    def __init__(self, model_name="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        
    def __call__(self, texts):
        # Handle both single text and batch
        if isinstance(texts, str):
            return self.embeddings.embed_query(texts)
        return self.embeddings.embed_documents(texts)
    
    def ndims(self):
        return 768  # nomic-embed-text uses 768 dimensions
    @staticmethod
    def SourceField():
        """Marks the source field for text"""
        return None
    
    @staticmethod
    def VectorField():
        """Marks the vector field for embeddings"""
        return Vector(768)

# Create the embedding function
embedding_func = OllamaEmbeddingFunction("nomic-embed-text")

class Document(LanceModel):
    id: str
    text: str = embedding_func.SourceField()
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_lancedb_table(db_path: str, table_name: str, overwrite: bool = True):
    db = lancedb.connect(db_path)
    mode = 'overwrite' if overwrite else 'create'
    table = db.create_table(table_name, schema=Document, mode=mode)
    table.create_fts_index('text', replace=overwrite)
    return table

def add_documents_to_table(table: LanceTable, knowledge_base_dir: str):
    docs = []
    knowledge_base = Path(knowledge_base_dir)

    for md_file in knowledge_base.glob('*.md'):
        print(f'Processing {md_file.name}')
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f'{md_file.stem}_{i}'
                docs.append({'id': doc_id, 'text': chunk})

    if docs:
        try:
            table.add(docs)
            print(f'Added {len(docs)} documents to the table.')
        except Exception as e:
            print(f'Error adding documents: {e}')
    else:
        print('No documents found.')

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
            doc_id = f'{Path(csv_path).stem}_{idx}_{i}'
            docs.append({'id': doc_id, 'text': chunk})
    
    if docs:
        table.add(docs)
        print(f'Added {len(docs)} CSV documents.')
    else:
        print('No CSV data added.')

def retrieve_similar_docs(
    table: LanceTable,
    query: str,
    query_type: str = 'hybrid',
    limit: int = 100,
    reranker_weight: float = 0.7
):
    reranker = LinearCombinationReranker(weight=reranker_weight)
    return (
        table.search(query, query_type=query_type)
        .rerank(reranker=reranker)
        .limit(limit)
        .to_list()
    )

def setup_lancedb():
    db_path = './db'
    table_name = 'knowledge'
    knowledge_base_dir = './knowledge-base'
    csv_path = './data/presupuesto_mexico__2020.csv'

    table = create_lancedb_table(db_path, table_name, overwrite=True)
    add_documents_to_table(table, knowledge_base_dir)
    add_csv_to_table(table, csv_path)
    return table

if __name__ == '__main__':
    table = setup_lancedb()
    results = retrieve_similar_docs(table, "education budget 2020")
    for doc in results[:3]:
        print(f"ID: {doc['id']}\nText: {doc['text'][:200]}...\n")