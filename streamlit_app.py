import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
BACKEND_HEALTH_URL = os.getenv("BACKEND_HEALTH_URL", "http://localhost:8000/health")
BACKEND_CONNECT_TIMEOUT = float(os.getenv("BACKEND_CONNECT_TIMEOUT", "5"))
BACKEND_READ_TIMEOUT = float(os.getenv("BACKEND_READ_TIMEOUT", "300"))

st.set_page_config(page_title="IC-Expert Agent", page_icon="🔬", layout="wide")
st.title("🔬 IC-Expert Agent")
st.caption("集成电路领域专业 ReAct Agent | LangGraph + LlamaIndex + RAGAS")

with st.sidebar:
    st.subheader("服务状态")
    st.caption(f"后端超时配置: connect={BACKEND_CONNECT_TIMEOUT}s, read={BACKEND_READ_TIMEOUT}s")
    try:
        health_resp = requests.get(BACKEND_HEALTH_URL, timeout=(BACKEND_CONNECT_TIMEOUT, 10))
        if health_resp.ok:
            st.success("后端已连接")
            st.caption(f"API: {BACKEND_HEALTH_URL}")
        else:
            st.error(f"后端异常: {health_resp.status_code}")
    except Exception as e:
        st.error("后端不可用")
        st.caption(str(e))

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("tool_info"):
            st.caption(f"🔧 {message['tool_info']}")
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("输入你的 IC 问题，例如：乘法器时序优化有哪些方法？"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端 FastAPI（流式）
    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        message_placeholder = st.empty()
        full_response = ""
        tool_info_list = []
        
        try:
            with requests.post(
                BACKEND_URL,
                json={"query": prompt},
                stream=True,
                timeout=(BACKEND_CONNECT_TIMEOUT, BACKEND_READ_TIMEOUT)
            ) as response:
                if not response.ok:
                    error_body = response.text
                    raise RuntimeError(f"HTTP {response.status_code}: {error_body}")
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk.get("type") == "answer":
                                    full_response += chunk["content"]
                                    message_placeholder.markdown(full_response + "▌")
                                elif chunk.get("type") == "tool_call":
                                    tool_info = chunk.get("content", "")
                                    if tool_info and tool_info not in tool_info_list:
                                        tool_info_list.append(tool_info)
                                        tool_placeholder.caption("🔧 " + " | ".join(tool_info_list))
                                elif chunk.get("type") == "error":
                                    full_response = f"❌ {chunk.get('content', '后端发生未知错误')}"
                                    message_placeholder.markdown(full_response)
                                    break
                            except Exception:
                                pass
        except requests.exceptions.ReadTimeout:
            full_response = (
                "❌ 后端响应超时。"
                "可在环境变量中增大 BACKEND_READ_TIMEOUT（例如 600），"
                "或降低模型复杂度。"
            )
        except Exception as e:
            full_response = f"❌ 连接后端失败: {e}"
        
        message_placeholder.markdown(full_response)
        if tool_info_list:
            tool_placeholder.caption("🔧 " + " | ".join(tool_info_list))
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "tool_info": " | ".join(tool_info_list),
        }
    )