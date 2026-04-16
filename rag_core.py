import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from llama_index_rag import get_llama_retriever   # ← 导入我们刚写的文件
from typing import TypedDict, Annotated, List
import operator
import re
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载环境变量 (API Key)
load_dotenv()
CHROMA_PATH = "./chroma_db"


def _resolve_embedding_device() -> str:
    # Allow explicit override via env (e.g. EMBEDDING_DEVICE=cpu in Docker)
    manual_device = os.getenv("EMBEDDING_DEVICE")
    if manual_device:
        return manual_device

    # Default: use MPS on Apple Silicon when available, otherwise CPU.
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
# ====================== 【混合模式：本地开发用云端，Docker 用本地 Ollama】 ======================
if os.getenv("USE_LOCAL_OLLAMA", "false").lower() == "true":
    # Docker 部署时走本地 Ollama
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        temperature=0.7,
        streaming=True,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    print("🚀 使用本地 Ollama (qwen2.5:7b)")
else:
    # 本地开发默认走云端（你原来的 qwen3.6-plus）
    llm = ChatOpenAI(
        model="qwen3.6-plus",          # ← 你云端模型名
        temperature=0.7,
        streaming=True,
        # 统一使用 OPENAI_API_BASE（兼容保留 OPENAI_BASE_URL）
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    )
    print("☁️ 使用云端模型 (qwen3.6-plus)")
# ====================== LangSmith 配置 ======================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "ic-expert-agent-infra-prod"

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
    """【推荐使用】LlamaIndex 优化后的检索函数（替换原来的 Chroma 版）"""
    retriever = get_llama_retriever()
    nodes = retriever.retrieve(query_text)
    # 转成 LangChain 兼容的 Document 格式，保持 Agent 完全不改
    from langchain_core.documents import Document
    docs = [Document(page_content=node.text, metadata=node.metadata) for node in nodes]
    return docs

def get_rag_response(question):
    """核心 RAG 生成函数（已切换为 LlamaIndex）"""
    docs = search_rag(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

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

    print(f"😉 正在思考... (上下文长度: {len(context_text)})")
    response = llm.invoke(prompt_text)

    return response.content, docs

# 测试代码入口
if __name__ == "__main__":
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
    """IC领域专业知识检索工具 - 已升级为 LlamaIndex 优化版"""
    docs = search_rag(query)   # ← 使用 LlamaIndex
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
    # LangSmith 自动追踪所有 LLM 调用、Tool 调用、Token 消耗
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _latest_user_query(messages: List) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _guess_clock_period_ns(text: str, default: float = 5.0) -> float:
    # Parse values like "2.5ns" / "2.5 ns" to auto-fill timing tool input.
    m = re.search(r"(\d+(?:\.\d+)?)\s*ns", text.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


def pre_tool_router(state: AgentState):
    """Hard guarantee: run one domain tool before LLM for each user turn."""
    query = _latest_user_query(state["messages"])
    if not query:
        return {"messages": []}

    q_lower = query.lower()
    tool_name = "ic_rag_search"

    verilog_markers = ["module", "endmodule", "always", "assign", "verilog", "rtl", "`timescale"]
    timing_markers = ["sdc", "时序", "约束", "clock", "setup", "hold", "多周期", "false path"]

    try:
        if any(k in q_lower for k in verilog_markers):
            tool_name = "verilog_code_analyzer"
            result = verilog_code_analyzer.invoke({"verilog_code": query})
        elif any(k in q_lower for k in timing_markers):
            tool_name = "timing_constraint_suggester"
            result = timing_constraint_suggester.invoke(
                {
                    "module_name": "user_module",
                    "clock_period_ns": _guess_clock_period_ns(query, 5.0),
                    "io_description": query,
                }
            )
        else:
            result = ic_rag_search.invoke({"query": query})

        injected = SystemMessage(
            content=(
                f"【预处理工具: {tool_name}】\n"
                f"以下是工具返回结果，请严格基于该结果作答：\n{result}"
            )
        )
        return {"messages": [injected]}
    except Exception as exc:
        # Keep agent available even if pre-tool fails.
        fallback = SystemMessage(content=f"【预处理工具失败】{exc}")
        return {"messages": [fallback]}

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
workflow.add_node("pre_tool", pre_tool_router)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", use_tool)

workflow.add_edge(START, "pre_tool")
workflow.add_edge("pre_tool", "llm")
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

    # 在 ReAct Agent 测试部分输入
    input_message = {"messages": [HumanMessage(content="乘法器时序优化有哪些方法？")]}
    result = app.invoke(input_message, {"recursion_limit": 20})
    print(result)