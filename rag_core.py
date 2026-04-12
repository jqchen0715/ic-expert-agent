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
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import tool
#新切分形式
from ic_text_splitter import ICCustomTextSplitter
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


def get_retriever(score_threshold: float = 0.5):
    """按需创建 retriever，避免全局连接长期占用数据库文件。"""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": score_threshold,
        },
    )


def search_rag(query_text: str):
    """IC领域检索函数（LangChain标准Retriever版）"""
    retriever = get_retriever(score_threshold=0.5)
    docs = retriever.invoke(query_text)   # ← 标准 LangChain 调用方式
    return docs

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


# ================== 保留你文件最上方的原有 RAG retriever 初始化代码 ==================

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

@tool
def ic_rag_search(query: str) -> str:
    """IC领域专业知识检索工具 - 任何关于Verilog、时序约束、PDK、芯片设计的问题都必须使用这个工具"""
    retriever = get_retriever(score_threshold=0.5)
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

tools = [ic_rag_search]
llm_with_tools = llm.bind_tools(tools)
   # ====================== 新增 Tool 2：Verilog 代码审查工具 ======================
@tool
def verilog_code_analyzer(verilog_code: str) -> str:
    """Verilog代码静态审查工具 - 检查latch、敏感列表、reset风格、blocking/non-blocking赋值等IC设计常见问题"""
    analysis = []
    
    # 1. 检查 always @(*) 是否完整（防止 latch）
    if "always @(*)" in verilog_code or "always_comb" not in verilog_code and "always @(" not in verilog_code:
        if any(kw in verilog_code for kw in ["if", "case", "else"]) and "else" not in verilog_code:
            analysis.append("⚠️ 可能存在 latch 推断（always块缺少完整else或default）")
    
    # 2. 检查敏感列表完整性
    if "always @" in verilog_code and "@(*)" not in verilog_code and "always_comb" not in verilog_code:
        analysis.append("⚠️ 建议使用 always_comb 或 always @(*)，避免不完整敏感列表导致综合后功能不一致")
    
    # 3. blocking vs non-blocking 赋值检查（简单版）
    if "<=" in verilog_code and "=" in verilog_code and "always @" in verilog_code:
        analysis.append("⚠️ 请确认 always 块中 <=（非阻塞）和 =（阻塞）使用是否正确（时序逻辑必须用 <=）")
    
    # 4. reset 风格检查
    if "rst" in verilog_code.lower() and "posedge" in verilog_code and "negedge" not in verilog_code:
        analysis.append("✅ 检测到异步复位风格（推荐）")
    elif "rst" in verilog_code.lower():
        analysis.append("⚠️ 检测到同步复位，建议确认是否为设计意图")
    
    if not analysis:
        analysis.append("✅ 代码通过基础静态审查，未发现明显IC设计常见问题")
    
    # 额外智能提示（你可以后续换成调用Ollama更深层分析）
    result = "\n".join(analysis)
    return f"【Verilog代码审查报告】\n{result}\n\n提示：如需更深度分析，请提供完整模块代码。"
 # ====================== 新增 Tool 3：时序约束建议工具 ======================
@tool
def timing_constraint_suggester(module_name: str, clock_period_ns: float, io_description: str = "") -> str:
    """时序约束建议工具 - 根据模块名、时钟周期、IO描述自动生成SDC格式约束"""
    sdc = f"""# ==================== 自动生成的SDC时序约束 ====================
# 模块: {module_name}
# 时钟周期: {clock_period_ns} ns ({1/clock_period_ns*1000:.1f} MHz)

# 1. 定义主时钟
create_clock -name sys_clk -period {clock_period_ns} [get_ports clk]

# 2. 时钟不确定性
set_clock_uncertainty 0.2 [get_clocks sys_clk]
set_clock_latency 0.1 [get_clocks sys_clk]

# 3. IO延迟约束（根据你提供的io_description智能调整）
"""
    # 简单根据描述生成IO约束
    if "input" in io_description.lower() or "in" in io_description.lower():
        sdc += f"set_input_delay -clock sys_clk -max {clock_period_ns*0.3:.2f} [get_ports {{输入端口列表}}]\n"
        sdc += f"set_input_delay -clock sys_clk -min {clock_period_ns*0.05:.2f} [get_ports {{输入端口列表}}]\n"
    
    if "output" in io_description.lower() or "out" in io_description.lower():
        sdc += f"set_output_delay -clock sys_clk -max {clock_period_ns*0.3:.2f} [get_ports {{输出端口列表}}]\n"
        sdc += f"set_output_delay -clock sys_clk -min {clock_period_ns*0.05:.2f} [get_ports {{输出端口列表}}]\n"
    
    sdc += "\n# 4. 其他常用约束（根据实际项目补充）\n"
    sdc += "set_false_path -from [get_ports rst_n]\n"
    sdc += "# set_multicycle_path -setup 2 -to [get_ports critical_output]\n"
    
    return sdc
# ====================== 更新 tools 列表和 System Prompt ======================
tools = [ic_rag_search, verilog_code_analyzer, timing_constraint_suggester]
llm_with_tools = llm.bind_tools(tools)

# 更强的 System Prompt（已包含全部3个Tool）
system_prompt = SystemMessage(content="""你是一个专业的IC设计专家Agent。
用户所有关于Verilog、时序约束、PDK、芯片设计、乘法器优化等问题，都必须先调用对应工具：
- 知识相关 → 必须先用 ic_rag_search
- Verilog代码审查 → 用 verilog_code_analyzer
- 时序约束/SDC → 用 timing_constraint_suggester

严禁直接用预训练知识回答IC专业问题，一定要先调用工具获取准确信息后再回答。""")

def call_llm(state: AgentState):
    # 每次都带上system prompt
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_map = {
            "ic_rag_search": ic_rag_search,
            "verilog_code_analyzer": verilog_code_analyzer,
            "timing_constraint_suggester": timing_constraint_suggester,
        }
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_impl = tool_map.get(tool_name)
            if not tool_impl:
                continue
            result = tool_impl.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        if tool_messages:
            return {"messages": tool_messages}
    return {"messages": []}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", use_tool)

workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("tool", "llm")

app = workflow.compile()

# 测试
if __name__ == "__main__":
    print("\n===== Tool 单测：verilog_code_analyzer =====")
    sample_verilog = """
module demo(input clk, input rst_n, input a, output reg y);
always @(clk or a) begin
  if (a) y = 1'b1;
end
endmodule
"""
    print(verilog_code_analyzer.invoke({"verilog_code": sample_verilog}))

    print("\n===== Tool 单测：timing_constraint_suggester =====")
    print(
        timing_constraint_suggester.invoke(
            {
                "module_name": "mul_top",
                "clock_period_ns": 2.5,
                "io_description": "input data_in, output data_out",
            }
        )
    )

    input_message = {"messages": [HumanMessage(content="帮我优化一个乘法器的时序约束")]}
    result = app.invoke(input_message, {"recursion_limit": 20})
    print(result)