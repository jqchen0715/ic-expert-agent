import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from dotenv import load_dotenv
# 1. 加载文档的工具
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# 2. 切分文本的工具
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 3. 向量数据库
from langchain_community.vectorstores import Chroma
# 4. 向量化模型 (用来把文字变成数字)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- 配置路径 ---
DATA_PATH = "./data"  # 存放 PDF 的文件夹
CHROMA_PATH = "./chroma_db"  # 存放向量数据库的文件夹 (自动生成)
embeddings = HuggingFaceEmbeddings(
        model_name="model/m3e-base",
        model_kwargs={'device': 'mps'}  # 关键：这行代码能调用你 Mac 的 GPU/NPU 加速
    )
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
    temperature=0.7,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
def create_vector_db():
    """
    核心逻辑：优先复用已有数据库；仅在不存在时执行构建
    """
    # --- 1. 优先复用已有数据库 ---
    if os.path.isdir(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"✅ 检测到已有数据库 {CHROMA_PATH}，跳过文档切分与重建。")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # --- 2. 检查原料 ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：没有找到 {DATA_PATH} 文件夹")
        return None

    # --- 3. 加载新文档 ---
    print("🔄 1. 正在加载 PDF 文档...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("❌ 错误：data 文件夹里没有 PDF！")
        return None
    print(f"   ✅ 成功加载 {len(documents)} 页文档。")

    # --- 4. 切分 ---
    print("🔄 2. 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ 成功切分为 {len(chunks)} 个文本块。")

    # --- 5. 向量化并存储 ---
    print("🔄 3. 正在写入新数据库...")
    # 仅在数据库不存在时会走到这里进行首次构建
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"   ✅ 向量数据库已重建完成！")
    return db


def search_rag(query_text):
    """
    测试检索功能：输入问题 -> 返回最相关的片段
    """
    # 重新加载已经保存的数据库
    embeddings = HuggingFaceEmbeddings(
        model_name="model/m3e-base",
        model_kwargs={'device': 'mps'}  # 关键：这行代码能调用你 Mac 的 GPU/NPU 加速
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print(f"\n🔍 正在检索问题：{query_text}")
    # k=3 表示返回最相关的 3 个片段
    results = db.similarity_search(query_text, k=3)

    return results

def get_rag_response(question):
    """
    核心逻辑：检索 + 生成
    """
    # 1. 连接数据库
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. 检索 (Retrieve) - 找 3 个相关片段
    docs = db.similarity_search(question, k=3)

    # 3. 拼接上下文 (Context)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 4. 构造 Prompt (这就是 Prompt Engineering！)
    # 我们强制要求它扮演 IC 专家，并且只能基于 Context 回答
    PROMPT_TEMPLATE = """
    你是一名集成电路(IC)领域的资深技术专家。请基于下面的【参考资料】回答用户的问题。

    规则：
    1. 如果参考资料里有答案，请用专业、简洁的语言回答。
    2. 如果参考资料里没有答案，请直接说“知识库中未找到相关信息”，不要瞎编。
    3. 如果涉及 Verilog 代码，请确保语法正确。

    【参考资料】：
    {context}

    【用户问题】：
    {question}
    """

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_text = prompt.format(context=context_text, question=question)

    # 5. 生成 (Generate)
    print(f"😉 正在思考... (Prompt 长度: {len(prompt_text)})")
    response = llm.invoke(prompt_text)

    # 返回：生成的回答 + 引用来源
    return response.content, docs


# 测试代码入口
if __name__ == "__main__":
    # 第一步：强制重新学习！(这一步就是你在图片里看到的逻辑)
    # 它会清空旧的 chroma_db，把 data 文件夹里的新 PDF 重新吃进去
    create_vector_db()

    # 第二步：学习完了，现在开始提问
    question = "什么是 Verilog？"
    answer, sources = get_rag_response(question)

    print(f"\n❓ 问题: {question}")
    print("-" * 50)
    print(f"😎 AI 回答: {answer}")
    print("-" * 50)
    print("📚 参考来源:")
    for doc in sources:
        print(f" - {doc.metadata.get('source')} (内容片段: {doc.page_content[:20]}...)")
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import tool

# ================== 保留你文件最上方的原有 RAG retriever 初始化代码 ==================
# （retriever = ... 那几行不要动）

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

llm = ChatOllama(model="qwen2.5:14b", temperature=0.7)

@tool
def ic_rag_search(query: str) -> str:
    """IC领域专业知识检索工具"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

tools = [ic_rag_search]
llm_with_tools = llm.bind_tools(tools)

def call_llm(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def use_tool(state: AgentState):
    """执行 Tool 并返回结果"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        if tool_call["name"] == "ic_rag_search":
            result = ic_rag_search.invoke(tool_call["args"])
            return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}
    return {"messages": []}

# ================== 真正的 ReAct 流程 ==================
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", use_tool)

workflow.add_edge(START, "llm")

# 条件边：如果 LLM 决定调用 Tool，就去 tool 节点，否则结束
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    return END

workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("tool", "llm")   # Tool 执行完再回到 LLM 继续思考

app = workflow.compile()

# 测试（直接运行这个）
if __name__ == "__main__":
    input_message = {"messages": [HumanMessage(content="帮我优化一个乘法器的时序约束")]}
    result = app.invoke(input_message)
    print(result)