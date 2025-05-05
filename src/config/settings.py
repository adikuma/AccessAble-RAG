from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
import os

def initialize_settings():
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    Settings.llm = OpenAI(
        api_key=api_key, 
        temperature=0, 
        model='gpt-4o'
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=api_key
    )
    return Settings
