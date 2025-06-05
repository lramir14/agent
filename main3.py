import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import ollama
import pypdf

# Constants
PERSIST_DIR = "./chroma_database"
COLLECTION_NAME = "RAGTutorial"

# Ensure persistence directory exists
os.makedirs(PERSIST_DIR, exist_ok=True)

def create_client_and_collection():
    # Using PersistentClient for automatic persistence
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection

def upload_batch(batch_texts, batch_ids):
    client, collection = create_client_and_collection()
    existing_ids = set(collection.get()["ids"])
    
    filtered_texts = []
    filtered_ids = []

    for text, doc_id in zip(batch_texts, batch_ids):
        if doc_id not in existing_ids:
            filtered_texts.append(text)
            filtered_ids.append(doc_id)

    if filtered_texts:
        collection.add(documents=filtered_texts, ids=filtered_ids)
        return f"Uploaded {len(filtered_texts)} new docs"
    return "No new documents to upload in this batch"

def load_csv_to_batches(file_path, batch_size=1000):
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(tqdm(reader, desc="Reading CSV")):
            row_dict = {col.strip(): val.strip() for col, val in zip(header, row)}
            row_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
            doc_id = f"{os.path.basename(file_path)}_row_{i}"
            rows.append((row_text, doc_id))
    return [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

def upload_csv_parallel(file_path, batch_size=1000, num_workers=4):
    batches = load_csv_to_batches(file_path, batch_size)
    print(f"Total batches to upload: {len(batches)}")

    # Note: ProcessPoolExecutor may cause persistence issues
    # Consider ThreadPoolExecutor if you encounter problems
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            texts, ids = zip(*batch)
            futures.append(executor.submit(upload_batch, list(texts), list(ids)))

        for f in tqdm(futures, desc="Uploading batches"):
            print(f.result())

def upload_csv_sample(file_path, max_rows=1000):
    client, collection = create_client_and_collection()
    existing_ids = set(collection.get()["ids"])
    
    if any(f"{os.path.basename(file_path)}_row_" in eid for eid in existing_ids):
        print(f"📄 {file_path} already uploaded. Skipping.")
        return

    texts, ids = [], []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            row_text = " | ".join(cell.strip() for cell in row if cell.strip())
            doc_id = f"{os.path.basename(file_path)}_row_{i}"
            if row_text and doc_id not in existing_ids:
                texts.append(row_text)
                ids.append(doc_id)

    if texts:
        collection.add(documents=texts, ids=ids)
        print(f"✅ Uploaded {len(texts)} rows from {file_path}")
    else:
        print(f"⚠️ No new rows uploaded from {file_path}")

def upload_pdf(file_path):
    client, collection = create_client_and_collection()
    existing_ids = set(collection.get()["ids"])
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        uploaded_pages = 0
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            doc_id = f"{os.path.basename(file_path)}_page_{i}"
            if doc_id not in existing_ids and text.strip():
                collection.add(documents=[text], ids=[doc_id])
                uploaded_pages += 1
        print(f"✅ Uploaded {uploaded_pages}/{total_pages} pages from {file_path}")

def ask_question(query):
    try:
        client, collection = create_client_and_collection()
        results = collection.query(query_texts=[query], n_results=10)

        context_docs = results["documents"][0]
        print("🔍 Retrieved documents from ChromaDB:")
        for i, doc in enumerate(context_docs):
            print(f"Document {i+1}: {doc[:300]}...")

        context_intro = (
            "You are provided with the following documents retrieved from a knowledge base. "
            "Use this information to answer the user's question accurately and completely."
        )

        messages = [{"role": "system", "content": context_intro}]
        messages += [{"role": "system", "content": doc} for doc in context_docs]
        messages.append({"role": "user", "content": query})

        response = ollama.chat(model="myqwen3", messages=messages)
        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    # Ensure persistence directory exists
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    # Upload sample data
    csv_file_path = "./data/mx_bud_2020.csv"
    upload_csv_sample(csv_file_path, max_rows=1000)

    # Upload PDFs
    pdf_files = ["./pearl-primer.pdf"]
    for pdf_file in pdf_files:
        upload_pdf(pdf_file)

    # Interactive Q&A
    print("You can now ask questions. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nAsk something: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        answer = ask_question(query)
        print(answer)