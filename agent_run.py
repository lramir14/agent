
import lancedb
from langchain_community.llms import Ollama
from lancedb_setup import setup_lancedb, retrieve_similar_docs
from typing import Optional, List

class LocalAgent:
    """Simplified agent class for Ollama"""
    def __init__(self, name: str, model: str, system_prompt: str):
        self.name = name
        self.llm = Ollama(model=model)
        self.system_prompt = system_prompt
    
    def run_sync(self, prompt: str, message_history: Optional[List] = None) -> dict:
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        response = self.llm.invoke(full_prompt)
        return {
            'data': response,
            'all_messages': message_history + [{'role': 'assistant', 'content': response}] 
                          if message_history else None
        }

def setup_knowledge_query_agent() -> LocalAgent:
    """Optimizes queries for document retrieval"""
    return LocalAgent(
        name='Query Optimizer',
        model='qwen3:0.6b',
        system_prompt="""
        Convert questions into concise search queries using key terms.
        Example:
        Input: "How much was spent on schools last year?"
        Output: "education budget 2023"
        """
    )

def setup_main_agent() -> LocalAgent:
    """Generates answers using retrieved context"""
    return LocalAgent(
        name='RAG Assistant',
        model='qwen3:0.6b',
        system_prompt="""
        Answer ONLY using the provided context. Be concise.
        If context is insufficient, say "I don't have enough information."
        """
    )

def main():
    # Initialize
    knowledge_table = setup_lancedb()
    knowledge_query_agent = setup_knowledge_query_agent()
    main_agent = setup_main_agent()
    message_history = None
    
    # Chat loop
    while True:
        query = input("\nEnter query (type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        
        # Step 1: Query optimization
        res = knowledge_query_agent.run_sync(query)
        knowledge_query = res['data']
        print(f"\nOptimized query: {knowledge_query}")
        
        # Step 2: Document retrieval
        retrieved_docs = retrieve_similar_docs(knowledge_table, knowledge_query, limit=5)
        
        # Step 3: Context filtering
        knowledge_context = "\n".join(
            doc['text'] for doc in retrieved_docs 
            if doc.get('_relevance_score', 0) > 0.5  # Adjust threshold as needed
        )
        
        # Step 4: Generate response
        prompt = f"Context:\n{knowledge_context}\n\nQuestion: {query}"
        response = main_agent.run_sync(prompt, message_history=message_history)
        
        print(f"\nAnswer: {response['data']}")
        message_history = response['all_messages']

if __name__ == "__main__":
    main()