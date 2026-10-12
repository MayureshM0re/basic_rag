from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# Point to local Ollama
Settings.llm = Ollama(
    model="qwen3.5:4b-q4_K_M",
    context_window=8192,
    request_timeout=300.0,
)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b-q8_0")
Settings.chunk_size = 512        # smaller chunks = less VRAM pressure

# Chroma in-memory (or persist_directory="./chroma_db" to save to disk)
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def build_index(docs_path: str = "./docs"):
    documents = SimpleDirectoryReader(docs_path).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    return index

def get_query_engine(index):
    return index.as_query_engine(similarity_top_k=3)