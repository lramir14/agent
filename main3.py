import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import ollama
import pypdf

# Constants
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "RAGTutorial"

def create_client_and_collection():
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection

def upload_batch(batch_texts, batch_ids):
    """
    Each process creates its own client and collection,
    then uploads its batch safely.
    """
    client, collection = create_client_and_collection()
    collection.add(documents=batch_texts, ids=batch_ids)
    return f"Uploaded batch of {len(batch_texts)} docs"

def load_csv_to_batches(file_path, batch_size=1000):
    """
    Load CSV rows, convert to text with 'key: value' pairs,
    split into batches for parallel upload.
    """
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(tqdm(reader, desc="Reading CSV")):
            row_dict = {col.strip(): val.strip() for col, val in zip(header, row)}
            row_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
            doc_id = f"{os.path.basename(file_path)}_row_{i}"
            rows.append((row_text, doc_id))
    # Split into batches
    batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
    return batches

def upload_csv_parallel(file_path, batch_size=1000, num_workers=4):
    batches = load_csv_to_batches(file_path, batch_size=batch_size)
    print(f"Total batches to upload: {len(batches)}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            texts, ids = zip(*batch)
            futures.append(executor.submit(upload_batch, list(texts), list(ids)))

        for f in tqdm(futures, desc="Uploading batches"):
            result = f.result()
            print(result)

def upload_pdf(file_path):
    """
    Upload PDF pages one by one into the collection.
    """
    client, collection = create_client_and_collection()
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            doc_id = f"{os.path.basename(file_path)}_page_{i}"
            collection.add(documents=[text], ids=[doc_id])
        print(f"Uploaded {total_pages} pages from {file_path}")

def ask_question(query):
    """
    Query ChromaDB and use ollama for answer generation.
    """
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

def upload_csv_sample(file_path, max_rows=1000):
    """
    Uploads up to `max_rows` rows from a CSV file into the ChromaDB collection.
    Each full row (joined string) is treated as a document.
    """
    client, collection = create_client_and_collection()
    texts, ids = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                row_text = " | ".join(cell.strip() for cell in row if cell.strip())
                if row_text:
                    texts.append(row_text)
                    ids.append(f"{os.path.basename(file_path)}_row_{i}")
            except Exception:
                continue

    collection.add(documents=texts, ids=ids)
    print(f"✅ Uploaded {len(texts)} rows from {file_path}")



if __name__ == "__main__":
    # Upload CSV data (parallelized)
    upload_csv_sample("./data/mx_bud_2020.csv", max_rows=1000)
    #csv_file_path = "./data/mx_bud_2020.csv"
    start_time = time.time()
    #upload_csv_parallel(csv_file_path, batch_size=1000, num_workers=4)
    end_time = time.time()
    #print(f"✅ CSV upload completed in {end_time - start_time:.2f} seconds.")

    # Upload PDF files (sequential, can add more files)
    pdf_files = ["./pearl-primer.pdf"]  #  "./ddp.textbook.pdf" change to your PDF files
    for pdf_file in pdf_files:
        print(f"Uploading PDF: {pdf_file}")
        upload_pdf(pdf_file)

    print("You can now ask multiple questions. Type 'exit' or 'quit' to end.")

    while True:
        query = input("\nPlease ask something: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = ask_question(query)
        print(answer)
