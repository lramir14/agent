import os
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import chromadb
import ollama
import pypdf
from typing import List, Tuple, Optional

class ChromaManager:
    def __init__(self, persist_dir: str = "./chroma_database", collection_name: str = "RAGTutorial"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def _create_doc_id(self, source: str, identifier: str) -> str:
        return f"{os.path.basename(source)}_{identifier}"

    def upload_csv(self, file_path: str, max_rows: Optional[int] = None) -> Tuple[int, int]:
        existing_ids = set(self.collection.get()["ids"])
        uploaded = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            texts, ids = [], []
            
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                
                row_text = " | ".join(f"{h}: {v.strip()}" for h, v in zip(header, row) if any(v.strip() for v in row) else None
                doc_id = self._create_doc_id(file_path, f"row_{i}")
                
                if row_text and doc_id not in existing_ids:
                    texts.append(row_text)
                    ids.append(doc_id)
            
            if texts:
                self.collection.add(documents=texts, ids=ids)
                uploaded = len(texts)
        
        return (i+1, uploaded)

    def upload_pdf(self, file_path: str) -> Tuple[int, int]:
        existing_ids = set(self.collection.get()["ids"])
        uploaded = 0
        
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                doc_id = self._create_doc_id(file_path, f"page_{i}")
                
                if text.strip() and doc_id not in existing_ids:
                    self.collection.add(documents=[text], ids=[doc_id])
                    uploaded += 1
        
        return (total, uploaded)

    def query(self, question: str, n_results: int = 5) -> Tuple[str, List[str]]:
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            context = "\n\n".join([
                f"SOURCE {i+1}:\n{text}"
                for i, text in enumerate(results["documents"][0])
            ])
            
            response = ollama.chat(
                model="myqwen3",
                messages=[
                    {"role": "system", "content": "Answer using ONLY the provided context"},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
            
            return response["message"]["content"], results["documents"][0]
        
        except Exception as e:
            return f"Error: {str(e)}", []