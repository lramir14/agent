from chroma_database.manager import ChromaManager

def main():
    db = ChromaManager()
    
    # Example usage
    print("Uploading sample data...")
    processed, uploaded = db.upload_csv("./data/mx_bud_2020.csv", max_rows=1000)
    print(f"CSV: Processed {processed} rows, uploaded {uploaded} new documents")
    
    total, uploaded = db.upload_pdf("./pearl-primer.pdf")
    print(f"PDF: {total} pages total, uploaded {uploaded} new pages")
    
    print("\nAsk questions (type 'exit' to quit):")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == 'exit':
            break
            
        answer, sources = db.query(query)
        print(f"\nAnswer: {answer}\n")
        print("Sources:")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source[:200]}...")

if __name__ == "__main__":
    main()