import ollama 
import chromadb
import pypdf
import csv 
import os 

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("RAGTutorial") #db table 

def upload_pdf(file_path): 
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        id=0
        
        for page in pdf_reader.pages:
            collection.add(
                documents = [page.extract_text()],
                ids=[f"{file_path}{id}"]
            )
            id +=1 
            
#upload_pdf("pearl-primer.pdf")
#upload_pdf("ddp.textbook.pdf")

ddef upload_csv(file_path, text_column=0, batch_size=100):
    texts = []
    ids = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header
        for i, row in enumerate(reader):
            try:
                text = row[text_column].strip()
                if text:
                    texts.append(text)
                    ids.append(f"{os.path.basename(file_path)}_row_{i}")

                    # Add in batches
                    if len(texts) >= batch_size:
                        collection.add(documents=texts, ids=ids)
                        texts.clear()
                        ids.clear()
            except IndexError:
                print(f"Skipping row {i}, no column {text_column}")

    # Add any remaining docs
    if texts:
        collection.add(documents=texts, ids=ids)


upload_csv("./data/mx_bud_2020.csv")

query = input(" Please ask something about your database>>>") 

def ask_question(query):
    closest_pages = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    context_docs = closest_pages["documents"][0]
    
    response = ollama.chat(
        model="qwen3:0.6b",
        messages=[
            {"role": "system", "content": doc} for doc in context_docs
        ] + [{"role": "user", "content": query}]
    )
    
    return response["message"]["content"]

# === CLI Interface ===
if __name__ == "__main__":
    query = input("Please ask something")
    print(ask_question(query))