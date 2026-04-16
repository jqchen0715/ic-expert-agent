# 1. 加载文档的工具
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# 3. 向量数据库
from langchain_community.vectorstores import Chroma
# 4. 向量化模型 (用来把文字变成数字)
from langchain_community.embeddings import HuggingFaceEmbeddings
#新切分形式
from ic_text_splitter import ICCustomTextSplitter
import os

# --- 配置路径 ---
DATA_PATH = "./data"  # 存放 PDF 的文件夹
CHROMA_PATH = "./chroma_db"  # 存放向量数据库的文件夹 (自动生成)


def _resolve_embedding_device() -> str:
    manual_device = os.getenv("EMBEDDING_DEVICE")
    if manual_device:
        return manual_device

    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


EMBEDDING_DEVICE = _resolve_embedding_device()
embeddings = HuggingFaceEmbeddings(
        model_name="model/m3e-base",
        model_kwargs={"device": EMBEDDING_DEVICE}
    )
print(f"🔧 Embedding device: {EMBEDDING_DEVICE}")
def create_vector_db():
    """核心逻辑：强制重建数据库 (加载 PDF -> 切分 -> 向量化 -> 存入数据库)。"""
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print(f"🗑️ 已删除旧数据库 {CHROMA_PATH}，正在准备重建...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：没有找到 {DATA_PATH} 文件夹")
        return None

    print("🔄 1. 正在加载 PDF 文档...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("❌ 错误：data 文件夹里没有 PDF！")
        return None
    print(f"   ✅ 成功加载 {len(documents)} 页文档。")

    print("🔄 2. 正在切分文本（IC定制化策略）...")
    # 替换为自定义IC分割器
    text_splitter = ICCustomTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)  # 用自定义的split_documents方法
    print(f"   ✅ 成功切分为 {len(chunks)} 个文本块。")

    print("🔄 3. 正在写入新数据库...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"   ✅ 向量数据库已重建完成！")
    return db
if __name__ == "__main__":
    create_vector_db()