import os
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import chromadb
import ollama
import pypdf
from typing import List, Tuple, Optional
import time
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from duckduckgo_search import DDGS

class ChromaManager:
    def __init__(self, persist_dir: str = "./chroma_database", collection_name: str = "RAGTutorial"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def _create_doc_id(self, source: str, identifier: str) -> str:
        return f"{os.path.basename(source)}_{identifier}"

    def upload_csv(self, file_path: str, max_rows: Optional[int] = None, batch_size: int = 1000) -> Tuple[int, int]:
        uploaded = 0
        processed = 0
        start_total = time.time()

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            texts, ids = [], []

            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break

                if not any(v.strip() for v in row):
                    continue

                row_text = " | ".join(f"{h}: {v.strip()}" for h, v in zip(header, row))
                doc_id = self._create_doc_id(file_path, f"row_{i}")

                texts.append(row_text)
                ids.append(doc_id)
                processed += 1

                if len(texts) >= batch_size:
                    start_batch = time.time()
                    self.collection.add(documents=texts, ids=ids)
                    print(f"Uploaded batch of {len(texts)} in {time.time() - start_batch:.2f}s")
                    uploaded += len(texts)
                    texts, ids = [], []

            if texts:
                start_batch = time.time()
                self.collection.add(documents=texts, ids=ids)
                print(f"Uploaded final batch of {len(texts)} in {time.time() - start_batch:.2f}s")
                uploaded += len(texts)

        print(f"\nTotal time: {time.time() - start_total:.2f}s")
        return (processed, uploaded)


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

    def search_web(self, query: str, num_results: int = 5):
        """Use DuckDuckGo to search the web and return result snippets."""
        with DDGS() as ddgs:
            results = ddgs.text(query, region='wt-wt', safesearch='moderate', max_results=num_results)
            return [r["body"] for r in results]
        
    def query_with_web(self, query: str):
        """Combine local RAG with web search context."""
        # Local search
        local_answer, local_sources = self.query(query)

        # Web search
        web_snippets = self.search_web(query)

        # Combine answer and sources
        final_answer = f"{local_answer}\n\n🌐 Web Insights:\n" + "\n".join(f"- {s}" for s in web_snippets[:3])
        combined_sources = local_sources + web_snippets

        return final_answer, [[s] for s in combined_sources]
