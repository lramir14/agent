from pathlib import Path
import pandas as pd
import tiktoken
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from lancedb.table import LanceTable  # assuming you're using the LanceTable from lancedb

openai_func = get_registry().get('openai').create(name='text-embedding-3-small')

class Document(LanceModel):
    """
    Defines the schema for documents to be stored in LanceDB table.
    """
    id: str
    text: str = openai_func.SourceField()
    vector: Vector(openai_func.ndims()) = openai_func.VectorField()

def chunk_text(text: str, max_tokens: int = 8192, encoding_name: str = 'cl100k_base'):
    """
    Chunk text into smaller sections to fit in a max token limit.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    for i in range(0, len(tokens), max_tokens):
        yield encoding.decode(tokens[i: i + max_tokens])

def create_lancedb_table(db_path: str, table_name: str, overwrite: bool = True):
    """
    Connect to LanceDB and create a table to store knowledge documents.
    """
    db = lancedb.connect(db_path)
    mode = 'overwrite' if overwrite else 'create'
    table = db.create_table(table_name, schema=Document, mode=mode)
    table.create_fts_index('text', replace=overwrite)
    return table

def drop_lancedb_table(db_path: str, table_name: str):
    """
    Drop a LanceDB table if it exists.
    """
    db = lancedb.connect(db_path)
    db.drop_table(table_name, ignore_missing=True)

def add_documents_to_table(table: LanceTable, knowledge_base_dir: str, max_tokens: int = 8192):
    """
    Add markdown docs from local directory to the LanceDB table.
    """
    docs = []
    knowledge_base = Path(knowledge_base_dir)

    for md_file in knowledge_base.glob('*.md'):
        print(f'Processing {md_file.name}')
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
            for i, chunk in enumerate(chunk_text(text, max_tokens=max_tokens)):
                doc_id = f'{md_file.stem}_{i}'
                docs.append({'id': doc_id, 'text': chunk})

    if docs:
        try:
            table.add(docs)
            print(f'Added {len(docs)} documents (chunks) to the table.')
        except Exception as e:
            print(f'Error adding documents: {e}')
    else:
        print('No documents found or added.')

def retrieve_similar_docs(table: LanceTable, query: str, query_type: str = 'hybrid', limit: int = 100, reranker_weight: float = 0.7):
    """
    Retrieve documents from LanceDB table using hybrid (semantic + full text) search and rerank.
    """
    reranker = LinearCombinationReranker(weight=reranker_weight)
    results = (
        table.search(query, query_type=query_type)
        .rerank(reranker=reranker)
        .limit(limit)
        .to_list()
    )
    return results

def add_csv_to_table(table: LanceTable, csv_path: str, max_tokens: int = 8192):
    """
    Load a CSV file, convert each row to a plain-text document, and add to LanceDB table.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    docs = []
    
    for idx, row in df.iterrows():
        row_dict = row.dropna().to_dict()
        text = "\n".join(f"{key}: {value}" for key, value in row_dict.items())
        
        for i, chunk in enumerate(chunk_text(text, max_tokens=max_tokens)):
            doc_id = f'{Path(csv_path).stem}_{idx}_{i}'
            docs.append({'id': doc_id, 'text': chunk})
    
    if docs:
        table.add(docs)
        print(f'Added {len(docs)} CSV-derived documents (chunks) to the table.')
    else:
        print('No rows found or added from CSV.')


def setup_lancedb():
    """
    Setup lancedb table with initial config 
    """
    db_path = './db'
    table_name = 'knowledge' 
    knowledge_base_dir = './knowledge-file'
    csv_path = './data/presupuesto_mexico__2020.csv'

    table = create_lancedb_table(db_path, table_name, overwrite=True)

    add_documents_to_table(table, knowledge_base_dir)
    add_csv_to_table(table, csv_path)  # this is called inside setup_lancedb()


if __name__ == '__main__':
    setup_lancedb()
