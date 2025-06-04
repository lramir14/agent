import os
# Block any OpenAI initialization attempts
os.environ["OPENAI_API_KEY"] = "local"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"  # Point to Ollama if anything tries to use it

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from lancedb_setup import setup_lancedb, retrieve_similar_docs
import warnings

# Suppress all OpenAI-related warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*openai.*")

class LocalRAGSystem:
    def __init__(self):
        print("Initializing local RAG system...")
        self.db = setup_lancedb()
        self.llm = Ollama(model="qwen3:0.6b")
        self._verify_setup()
        
        self.query_prompt = ChatPromptTemplate.from_template(
            "Extract key search terms: {question}\nQuery:"
        )
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

    def _verify_setup(self):
        """Verify everything is working locally"""
        try:
            test_vec = self.db._embedding_function("test")
            assert len(test_vec) == 768, "Invalid embedding dimension"
        except Exception as e:
            raise RuntimeError(f"Setup verification failed: {str(e)}")

    def generate_response(self, question: str):
        # Generate search query
        search_query = (self.query_prompt | self.llm).invoke(
            {"question": question}
        ).strip()
        
        print(f"Searching for: {search_query}")
        
        # Retrieve documents (force local-only)
        docs = retrieve_similar_docs(
            self.db,
            search_query,
            query_type="vector"  # Disable hybrid search
        )
        
        context = "\n".join(d['text'] for d in docs)
        answer = (self.answer_prompt | self.llm).invoke({
            "question": question,
            "context": context
        })
        
        return {
            "question": question,
            "answer": answer,
            "docs": docs
        }

def main():
    try:
        rag = LocalRAGSystem()
        print("System ready. Type 'quit' to exit.")
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ('quit', 'exit'):
                break
                
            result = rag.generate_response(question)
            print(f"\nAnswer: {result['answer']}")
            if result['docs']:
                print("\nSources used:")
                for i, doc in enumerate(result['docs'], 1):
                    print(f"[{i}] {doc['text'][:150]}...")
            else:
                print("\nNo relevant documents found")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Run 'ollama serve' in another terminal")
        print("2. Verify models: 'ollama list' should show qwen3:0.6b and nomic-embed-text")
        print("3. Test embeddings directly with:")
        print("   python -c \"from langchain_community.embeddings import OllamaEmbeddings; print(len(OllamaEmbeddings(model='nomic-embed-text').embed_query('test')))\"")

if __name__ == "__main__":
    main()