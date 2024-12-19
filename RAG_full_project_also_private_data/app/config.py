import os
from dotenv import load_dotenv
from langchain_nomic import NomicEmbeddings
from app.utils.rec_fusion import reciprocal_rank_fusion
from app.utils.multi_query import get_unique_union
from langchain_openai import OpenAIEmbeddings

# Load the keys
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '../keys/.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
nomic_api_key = os.getenv("NOMIC_API_KEY")

# LLM Model Configuration
LLM_MODEL = "llama-3.1-8b-instant"

# Chunk Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings
EMBEDDING = NomicEmbeddings(model="nomic-embed-text-v1.5")

# retrival technique
RETRIVAL_TECHNIQUE = get_unique_union


# Templates
# Question template
QUERY_GENERATION_TEMPLATE = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines, give only questions and nothing else. Original question: {question}"""

# Answer template
FINAL_ANSWER_TEMPLATE = """Answer the following question in detail and provide its source of information based on the given context.

{context}

Question: {question}

Additionally, provide your confidence level using cosine similarity in your answer between 0-100% and please do not give explanation regarding this, give just confidence level.
"""