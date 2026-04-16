from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="model/m3e-base",
    model_kwargs={'device': 'mps'}
)
# ================== 保留的原有初始化代码 ==================
CHROMA_PATH = "chroma_db"  # 你的数据库路径，如果不同请改成你原来的
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5,
    },
)

# ================== 简化后的检索函数（给Agent Tool使用） ==================
def search_rag(query_text: str):
    """IC领域检索函数（LangChain标准Retriever版）"""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    # 每次都重新创建 retriever，保证最新配置
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",   # 只返回相似度够高的结果
        search_kwargs={
            "k": 5,                    # 返回最多5条
            "score_threshold": 0.75,   # 相似度阈值（可自行调高/低）
        }
    )
    docs = retriever.invoke(query_text)   # ← 标准 LangChain 调用方式
    return docs

# ================== LangGraph ReAct Agent ==================
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "qwen-plus"),
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

@tool
def ic_rag_search(query: str) -> str:
    """IC领域专业知识检索工具 - 任何关于Verilog、时序约束、PDK、芯片设计的问题都必须使用这个工具"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

tools = [ic_rag_search]
llm_with_tools = llm.bind_tools(tools)

# 强 System Prompt（关键！）
system_prompt = SystemMessage(content="""你是一个IC设计专家Agent。
用户所有关于Verilog、时序约束、PDK、芯片设计、乘法器优化等问题，都必须先调用ic_rag_search工具检索专业文档后再回答。
不要直接用你的预训练知识回答IC专业问题，一定要先调用工具。""")

def call_llm(state: AgentState):
    # 每次都带上system prompt
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def use_tool(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        if tool_call["name"] == "ic_rag_search":
            result = ic_rag_search.invoke(tool_call["args"])
            return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}
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
    input_message = {"messages": [HumanMessage(content="帮我优化一个乘法器的时序约束")]}
    result = app.invoke(input_message, {"recursion_limit": 20})
    print(result)

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
    
    # 额外智能提示（你可以后续换成调用更强模型做深层分析）
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