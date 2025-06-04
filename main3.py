import ollama 
import chromadb
import pypdf
import csv 
import os 
import time 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("RAGTutorial")

# Upload PDF function (unchanged)
def upload_pdf(file_path): 
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            collection.add(
                documents=[page.extract_text()],
                ids=[f"{file_path}_page_{i}"]
            )

# Parallel upload function
def upload_csv(file_path, text_column=0, batch_size=5000, num_workers=4):
    texts = []
    ids = []
    futures = []
    
    # Estimate total rows for tqdm
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # minus header

    executor = ThreadPoolExecutor(max_workers=num_workers)

    def upload_batch(batch_texts, batch_ids):
        collection.add(documents=batch_texts, ids=batch_ids)

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for i, row in enumerate(tqdm(reader, total=total_lines, desc="Uploading CSV")):
            try:
                text = row[text_column].strip()
                if text:
                    texts.append(text)
                    ids.append(f"{os.path.basename(file_path)}_row_{i}")
                    if len(texts) >= batch_size:
                        futures.append(executor.submit(upload_batch, texts.copy(), ids.copy()))
                        texts.clear()
                        ids.clear()
            except IndexError:
                continue

        if texts:
            futures.append(executor.submit(upload_batch, texts.copy(), ids.copy()))

    # Wait for all threads to finish
    for future in futures:
        future.result()

    executor.shutdown()

# Ask a question
def ask_question(query):
    closest_pages = collection.query(
        query_texts=[query],
        n_results=3
    )
    context_docs = closest_pages["documents"][0]
    response = ollama.chat(
        model="qwen3:0.6b",
        messages=[{"role": "system", "content": doc} for doc in context_docs] + [{"role": "user", "content": query}]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    start_time = time.time()
    upload_csv("./data/mx_bud_2020.csv", batch_size=5000, num_workers=4)
    end_time = time.time()
    print(f"✅ Upload completed in {end_time - start_time:.2f} seconds.")

    query = input("Please ask something: ")
    print(ask_question(query))
