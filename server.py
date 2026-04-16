import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import app as agent_workflow
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv
from typing import Any
import re

load_dotenv()

app = FastAPI(title="IC-Expert Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 新增 Pydantic 请求模型 ======================
class ChatRequest(BaseModel):
    query: str


def _friendly_error_message(exc: Exception) -> str:
    text = str(exc)
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    # Common local Ollama failure: model not pulled on host machine.
    if "model" in text and "not found" in text:
        return (
            f"本地 Ollama 未找到模型 {model_name}。"
            f"请在宿主机执行: ollama pull {model_name}，并确认 OLLAMA_BASE_URL 可访问。"
        )
    return f"后端推理失败: {text}"


def _normalize_chunk_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def _split_for_streaming(text: str, chunk_size: int = 24):
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _extract_pretool_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"【预处理工具:\s*([^】]+)】", text)
    return m.group(1).strip() if m else ""

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    
    async def event_stream():
        try:
            inputs = {"messages": [HumanMessage(content=query)]}
            emitted_any_answer = False
            final_fallback_answer = ""
            emitted_tools = set()

            async for event in agent_workflow.astream_events(
                inputs,
                version="v2",
                config={"recursion_limit": 30},
            ):
                event_name = event.get("event")
                data = event.get("data", {})

                if event_name == "on_tool_start":
                    tool_name = event.get("name") or "unknown_tool"
                    if tool_name not in emitted_tools:
                        emitted_tools.add(tool_name)
                        yield f"data: {json.dumps({'type': 'tool_call', 'content': f'正在调用工具: {tool_name}'}, ensure_ascii=False)}\n\n"
                    continue

                if event_name == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    content = _normalize_chunk_content(getattr(chunk, "content", ""))
                    if content:
                        emitted_any_answer = True
                        for piece in _split_for_streaming(content):
                            yield f"data: {json.dumps({'type': 'answer', 'content': piece}, ensure_ascii=False)}\n\n"
                    continue

                # Fallback path: some providers only emit final chain chunk.
                if event_name in {"on_chain_stream", "on_chain_end"}:
                    chunk_payload = data.get("chunk") if isinstance(data, dict) else None
                    if isinstance(chunk_payload, dict):
                        for node_value in chunk_payload.values():
                            if isinstance(node_value, dict) and "messages" in node_value:
                                messages = node_value.get("messages") or []
                                if messages:
                                    for msg in messages:
                                        content = _normalize_chunk_content(getattr(msg, "content", ""))
                                        tool_name = _extract_pretool_name(content)
                                        if tool_name and tool_name not in emitted_tools:
                                            emitted_tools.add(tool_name)
                                            yield f"data: {json.dumps({'type': 'tool_call', 'content': f'本轮使用工具: {tool_name}'}, ensure_ascii=False)}\n\n"
                                    last_msg = messages[-1]
                                    content = _normalize_chunk_content(getattr(last_msg, "content", ""))
                                    if content:
                                        final_fallback_answer = content

            if not emitted_any_answer and final_fallback_answer:
                for piece in _split_for_streaming(final_fallback_answer):
                    yield f"data: {json.dumps({'type': 'answer', 'content': piece}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_payload = {
                "type": "error",
                "content": _friendly_error_message(exc),
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
async def health():
    use_local_ollama = os.getenv("USE_LOCAL_OLLAMA", "false").lower() == "true"
    active_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b") if use_local_ollama else "qwen3.6-plus"
    return {
        "status": "ok",
        "mode": "local-ollama" if use_local_ollama else "cloud-openai-compatible",
        "model": f"{active_model} + LangGraph + LlamaIndex",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)