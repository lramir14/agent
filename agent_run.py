
import lancedb
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from lancedb_setup import setup_lancedb, retrieve_similar_docs


def setup_knowledge_query_agent():
    """
    Set up the knowledge query agent.
    """
    agent = Agent(
        name='Knowledge Query Agent',
        model=OpenAIModel('gpt-4o-mini'),
        degc_type=str,
        result_type=str,
        system_prompt='From the input text string, please generate a query string to pass to the knowledge base.',
    )
    return agent

def setup_main_agent():
    """
    Set up the main agent.
    """
    agent = Agent(
        name='Main Agent',
        model=OpenAIModel('gpt-4o-mini'),
        system_prompt='You are a helpful assistant',
    )
    return agent