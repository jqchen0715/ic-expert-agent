import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from ic_text_splitter import ICCustomTextSplitter

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# ====================== LlamaIndex 配置 ======================
if os.getenv("USE_LOCAL_OLLAMA", "false").lower() == "true":
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=0.7,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    print("🚀 LlamaIndex 使用本地 Ollama (qwen2.5:7b)")
else:
    Settings.llm = OpenAI(
        model="qwen3.6-plus",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    print("☁️ LlamaIndex 使用云端模型 (qwen3.6-plus)")

Settings.embed_model = HuggingFaceEmbedding(model_name="model/m3e-base")
# ====================== 优化版 RAG（带 Metadata 过滤） ======================
def create_llama_index():
    """使用 LlamaIndex 重新构建索引（支持 Metadata 过滤）"""
    print("🔄 LlamaIndex 正在重建优化 RAG...")

    # 1. 加载文档 + IC 定制切分
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = ICCustomTextSplitter(chunk_size=800, chunk_overlap=100)
    lc_docs = text_splitter.split_documents(documents)
    nodes = [
        Document(text=doc.page_content, metadata=(doc.metadata or {}))
        for doc in lc_docs
    ]

    # 2. Chroma 作为向量存储（复用你原来的 chroma_db）
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("ic_expert")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. 构建索引（带 Metadata）
    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ LlamaIndex 优化 RAG 构建完成！支持 Metadata 过滤")
    return index

def get_llama_retriever():
    """支持 Metadata 过滤的 Retriever（例如只检索 Verilog 相关文档）"""
    index = create_llama_index() if not os.path.exists(CHROMA_PATH) else None
    # 如果已存在则直接加载
    if index is None:
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            chroma_collection = chroma_client.get_collection("ic_expert")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
        except Exception:
            # 目录存在但目标 collection 不存在时，自动重建一次索引。
            index = create_llama_index()
    
    return index.as_retriever(
        similarity_top_k=5,
        filters=None  # 后面可以加 MetadataFilter（如 {"source_type": "verilog"}）
    )

# 测试用
if __name__ == "__main__":
    retriever = get_llama_retriever()
    nodes = retriever.retrieve("乘法器时序优化")
    for node in nodes:
        print(node.text[:200] + "...")