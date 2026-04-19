import streamlit as st
import requests
import json
import time
import os
import threading
from datetime import datetime
from uuid import uuid4

# 后端 API 的地址
API_URL = "http://127.0.0.1:8000/chat"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 180
TYPEWRITER_STEP = 6
TYPEWRITER_DELAY = 0.006
UI_POLL_INTERVAL = 0.8
UI_POLL_INTERVAL_MS = int(UI_POLL_INTERVAL * 1000)
HISTORY_FILE = "chat_history.json"
DEFAULT_SESSION_TITLE = "未命名会话"


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def create_empty_session(title=None):
    now = _now_iso()
    return {
        "id": uuid4().hex[:8],
        "title": (title or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def _normalize_messages(messages):
    normalized = []
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if role in {"user", "assistant"} and isinstance(content, str):
            item = {"role": role, "content": content}
            if role == "assistant" and isinstance(msg.get("tools"), list):
                item["tools"] = msg.get("tools")
            normalized.append(item)
    return normalized


def generate_title_from_text(text, max_len=16):
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return DEFAULT_SESSION_TITLE

    # 优先按常见中文/英文断句符截断，让标题更自然。
    for sep in ["。", "？", "！", "?", "!", "\n"]:
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break

    cleaned = cleaned.strip(" ,，.。:：;；、\"'`()[]{}")
    if not cleaned:
        return DEFAULT_SESSION_TITLE

    if len(cleaned) > max_len:
        return cleaned[:max_len] + "..."
    return cleaned


def auto_title_for_messages(messages):
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            title = generate_title_from_text(msg.get("content", ""))
            return title or DEFAULT_SESSION_TITLE
    return DEFAULT_SESSION_TITLE


def load_all_sessions():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    sessions = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        sid = raw.get("id")
        if not isinstance(sid, str) or not sid:
            sid = uuid4().hex[:8]
        created_at = raw.get("created_at") if isinstance(raw.get("created_at"), str) else _now_iso()
        updated_at = raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else created_at
        messages = _normalize_messages(raw.get("messages", []))
        loaded_title = (raw.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE
        if loaded_title == DEFAULT_SESSION_TITLE and messages:
            loaded_title = auto_title_for_messages(messages)

        sessions.append(
            {
                "id": sid,
                "title": loaded_title,
                "created_at": created_at,
                "updated_at": updated_at,
                "messages": messages,
            }
        )
    return sessions


def save_all_sessions(sessions):
    payload = []
    for s in sessions:
        payload.append(
            {
                "id": s.get("id"),
                "title": (s.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
                "created_at": s.get("created_at", _now_iso()),
                "updated_at": s.get("updated_at", _now_iso()),
                "messages": _normalize_messages(s.get("messages", [])),
            }
        )

    temp_file = HISTORY_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, HISTORY_FILE)


@st.cache_resource
def get_history_lock():
    return threading.Lock()


def get_session_by_id(sessions, session_id):
    for s in sessions:
        if s.get("id") == session_id:
            return s
    return None


def persist_active_session_messages():
    persist_session_messages(st.session_state.active_session_id, st.session_state.messages)


def persist_session_messages(session_id, messages):
    with get_history_lock():
        sessions = load_all_sessions()
        target = get_session_by_id(sessions, session_id)
        if target is None:
            target = create_empty_session()
            target["id"] = session_id
            sessions.append(target)
        target["messages"] = _normalize_messages(messages)
        current_title = (target.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE
        if current_title == DEFAULT_SESSION_TITLE and target["messages"]:
            target["title"] = auto_title_for_messages(target["messages"])
        target["updated_at"] = _now_iso()
        save_all_sessions(sessions)


def append_assistant_message_to_session(session_id, content, tool_calls=None):
    with get_history_lock():
        sessions = load_all_sessions()
        target = get_session_by_id(sessions, session_id)
        if target is None:
            target = create_empty_session()
            target["id"] = session_id
            sessions.append(target)
        target.setdefault("messages", []).append(
            {
                "role": "assistant",
                "content": content,
                "tools": tool_calls or [],
            }
        )
        current_title = (target.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE
        if current_title == DEFAULT_SESSION_TITLE and target["messages"]:
            target["title"] = auto_title_for_messages(target["messages"])
        target["updated_at"] = _now_iso()
        save_all_sessions(sessions)


def session_preview_text(session, limit=24):
    rounds = build_chat_rounds(session.get("messages", []))
    if rounds:
        return short_text(rounds[-1].get("user", ""), limit=limit)
    return "空会话"


def session_display_label(session):
    title = ((session or {}).get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE
    preview = session_preview_text(session or {})
    if preview == "空会话":
        return title
    return f"{title} · {preview}"


def sort_sessions_by_recent(sessions):
    def _parse_ts(value):
        if not isinstance(value, str):
            return datetime.min
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min

    return sorted(sessions, key=lambda s: _parse_ts(s.get("updated_at")), reverse=True)


def rename_session(session_id, new_title):
    title = (new_title or "").strip()
    if not title:
        return False
    with get_history_lock():
        sessions = load_all_sessions()
        target = get_session_by_id(sessions, session_id)
        if target is None:
            return False
        target["title"] = title
        target["updated_at"] = _now_iso()
        save_all_sessions(sessions)
    return True


def delete_session(session_id):
    with get_history_lock():
        sessions = load_all_sessions()
        remaining = [s for s in sessions if s.get("id") != session_id]
        if not remaining:
            remaining = [create_empty_session()]
        save_all_sessions(remaining)
        return remaining


def clear_all_sessions():
    with get_history_lock():
        fresh = [create_empty_session()]
        save_all_sessions(fresh)
        return fresh


@st.cache_resource
def get_task_store():
    return {"lock": threading.Lock(), "tasks": {}}


def _update_task(task_id, **fields):
    store = get_task_store()
    with store["lock"]:
        task = store["tasks"].get(task_id)
        if task is None:
            return None
        task.update(fields)
        task["updated_at"] = _now_iso()
        return dict(task)


def _pop_task(task_id):
    store = get_task_store()
    with store["lock"]:
        return store["tasks"].pop(task_id, None)


def get_running_task_for_session(session_id):
    store = get_task_store()
    with store["lock"]:
        for tid, task in store["tasks"].items():
            if task.get("session_id") == session_id and task.get("status") == "running":
                snapshot = dict(task)
                snapshot["task_id"] = tid
                return snapshot
    return None


def run_generation_task(task_id):
    store = get_task_store()
    with store["lock"]:
        task = dict(store["tasks"].get(task_id) or {})
    if not task:
        return

    session_id = task.get("session_id")
    prompt = task.get("prompt", "")
    full_response = ""
    tool_calls = []
    generation_started = False
    received_event = False

    try:
        with requests.post(
            API_URL,
            json={"query": prompt},
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        ) as response:
            if not response.ok:
                msg = f"服务器报错: HTTP {response.status_code} - {response.text}"
                append_assistant_message_to_session(session_id, f"❌ {msg}")
                _update_task(task_id, status="failed", error=msg, stage="完成")
                _pop_task(task_id)
                return

            _update_task(task_id, stage="检索中")
            for event in iter_sse_events(response):
                received_event = True
                event_type = event.get("type", "")
                content = event.get("content", "")

                if event_type == "tool_call":
                    if content and content not in tool_calls:
                        tool_calls.append(content)
                        _update_task(task_id, tool_calls=tool_calls, stage="工具调用")
                elif event_type == "answer":
                    if not generation_started:
                        generation_started = True
                        _update_task(task_id, stage="生成中")
                    full_response += content
                    _update_task(task_id, full_response=full_response)
                elif event_type == "error":
                    msg = content or "后端返回未知错误"
                    append_assistant_message_to_session(session_id, f"❌ {msg}", tool_calls=tool_calls)
                    _update_task(task_id, status="failed", error=msg, stage="完成")
                    _pop_task(task_id)
                    return

        if full_response:
            append_assistant_message_to_session(session_id, full_response, tool_calls=tool_calls)
            _update_task(task_id, status="completed", stage="完成")
            _pop_task(task_id)
            return

        if not received_event:
            msg = "本轮未收到后端事件，可能检索过慢或后端阻塞。"
        else:
            msg = "本轮未生成有效回答，请重试或把问题描述得更具体。"
        append_assistant_message_to_session(session_id, f"⚠️ {msg}", tool_calls=tool_calls)
        _update_task(task_id, status="failed", error=msg, stage="完成")
        _pop_task(task_id)
    except requests.exceptions.ReadTimeout:
        msg = f"后端读取超时（>{READ_TIMEOUT}s），请稍后重试。"
        append_assistant_message_to_session(session_id, f"❌ {msg}", tool_calls=tool_calls)
        _update_task(task_id, status="failed", error=msg, stage="完成")
        _pop_task(task_id)
    except requests.exceptions.ConnectTimeout:
        msg = f"连接后端超时（>{CONNECT_TIMEOUT}s），请检查服务状态。"
        append_assistant_message_to_session(session_id, f"❌ {msg}", tool_calls=tool_calls)
        _update_task(task_id, status="failed", error=msg, stage="完成")
        _pop_task(task_id)
    except Exception as exc:
        msg = f"连接失败: {str(exc)}"
        append_assistant_message_to_session(session_id, f"❌ {msg}", tool_calls=tool_calls)
        _update_task(task_id, status="failed", error=msg, stage="完成")
        _pop_task(task_id)


def start_generation_task(session_id, prompt):
    task_id = uuid4().hex
    task = {
        "session_id": session_id,
        "prompt": prompt,
        "status": "running",
        "stage": "检索中",
        "tool_calls": [],
        "full_response": "",
        "error": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    store = get_task_store()
    with store["lock"]:
        store["tasks"][task_id] = task

    t = threading.Thread(target=run_generation_task, args=(task_id,), daemon=True)
    t.start()
    return task_id


def iter_sse_events(response):
    """从 text/event-stream 响应中迭代解析 data 事件。"""
    for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue

        payload = raw_line[6:]
        if payload == "[DONE]":
            break

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        yield event


def build_chat_rounds(messages):
    """将消息序列整理为用户-助手对话轮次。"""
    rounds = []
    pending_user = None

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            pending_user = msg
        elif role == "assistant" and pending_user:
            rounds.append(
                {
                    "user": pending_user.get("content", ""),
                    "assistant": msg.get("content", ""),
                    "tools": msg.get("tools", []),
                }
            )
            pending_user = None

    return rounds


def short_text(text, limit=28):
    one_line = (text or "").replace("\n", " ").strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def render_stage_status(slot, stage):
    slot.caption(f"⏱️ 阶段状态：{stage}")


def render_stage(stage_slot, message_slot, stage):
    """统一渲染阶段状态和对应提示，避免状态与文案不一致。"""
    render_stage_status(stage_slot, stage)
    if stage == "检索中":
        message_slot.markdown("🔄 正在检索数据手册...")
    elif stage == "工具调用":
        message_slot.markdown("🧰 正在调用工具...")
    elif stage == "生成中":
        message_slot.markdown("✍️ 正在生成回答...")


def append_with_typewriter(slot, full_text, chunk):
    """前端打字机效果：把一段文本按小步增量渲染。"""
    if not chunk:
        return full_text

    for i in range(0, len(chunk), TYPEWRITER_STEP):
        full_text += chunk[i:i + TYPEWRITER_STEP]
        slot.markdown(full_text + "▌")
        time.sleep(TYPEWRITER_DELAY)

    return full_text


st.markdown(
    """
<style>
section[data-testid="stSidebar"] .stButton > button {
    text-align: left;
    border-radius: 10px;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stTextInput input {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- 页面配置 ---
st.set_page_config(page_title="IC-Expert 芯片助手", layout="wide")

st.title("🧐 IC-Expert: 集成电路专业知识库助手")


# --- 主界面：聊天记录（本地持久化） ---
all_sessions = load_all_sessions()
if not all_sessions:
    all_sessions = [create_empty_session()]
    save_all_sessions(all_sessions)

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = all_sessions[-1]["id"]

active = get_session_by_id(all_sessions, st.session_state.active_session_id)
if active is None:
    st.session_state.active_session_id = all_sessions[-1]["id"]
    active = all_sessions[-1]

st.session_state.messages = active.get("messages", [])
st.session_state._messages_session_id = st.session_state.active_session_id

# --- 侧边栏：历史对话回顾 ---
with st.sidebar:
    st.divider()
    st.subheader("🕘 历史对话回顾（本地）")

    if st.button("➕ 新建会话", use_container_width=True):
        sessions = load_all_sessions()
        new_session = create_empty_session()
        sessions.append(new_session)
        save_all_sessions(sessions)
        st.session_state.active_session_id = new_session["id"]
        st.session_state.messages = []
        st.session_state._messages_session_id = new_session["id"]
        st.rerun()

    sessions_for_pick = sort_sessions_by_recent(load_all_sessions())

    st.caption("会话列表（最近优先）")
    list_container = st.container(height=300)
    with list_container:
        for sess in sessions_for_pick:
            sid = sess.get("id")
            if not sid:
                continue
            label = session_display_label(sess)
            if st.session_state.active_session_id == sid:
                label = "● " + label
            if st.button(
                label,
                key=f"pick_{sid}",
                use_container_width=True,
                type="primary" if st.session_state.active_session_id == sid else "secondary",
            ):
                st.session_state.active_session_id = sid
                st.session_state.messages = sess.get("messages", [])
                st.session_state._messages_session_id = sid
                st.rerun()

    with st.expander("会话管理", expanded=False):
        current_session = get_session_by_id(sessions_for_pick, st.session_state.active_session_id) or {}
        current_title = (current_session.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE
        rename_value = st.text_input("会话标题", value=current_title, key="rename_title_input")
        if st.button("💾 保存标题", use_container_width=True):
            if rename_session(st.session_state.active_session_id, rename_value):
                st.success("会话标题已更新")
                st.rerun()
            else:
                st.warning("标题不能为空")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 删除当前", use_container_width=True):
                remaining = delete_session(st.session_state.active_session_id)
                st.session_state.active_session_id = remaining[-1]["id"]
                st.session_state.messages = remaining[-1].get("messages", [])
                st.session_state._messages_session_id = remaining[-1]["id"]
                st.rerun()
        with c2:
            if st.button("🧹 清空全部", use_container_width=True):
                fresh = clear_all_sessions()
                st.session_state.active_session_id = fresh[0]["id"]
                st.session_state.messages = []
                st.session_state._messages_session_id = fresh[0]["id"]
                st.rerun()

    chat_rounds = build_chat_rounds(st.session_state.messages)
    if not chat_rounds:
        st.caption("暂无历史对话")
    else:
        selected_round = st.selectbox(
            "选择轮次",
            options=list(range(len(chat_rounds), 0, -1)),
            format_func=lambda i: f"第 {i} 轮 · {short_text(chat_rounds[i - 1]['user'])}",
        )

        selected = chat_rounds[selected_round - 1]
        with st.expander("查看本轮详情", expanded=True):
            st.markdown("**用户问题**")
            st.markdown(selected["user"] or "-")
            st.markdown("**助手回答**")
            st.markdown(selected["assistant"] or "-")
            tools = selected.get("tools") or []
            if tools:
                st.caption("🛠️ 本轮工具: " + " | ".join(tools))

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("tools"):
            st.caption("🛠️ 本轮工具: " + " | ".join(message["tools"]))
        st.markdown(message["content"])

running_task = get_running_task_for_session(st.session_state.active_session_id)
if running_task:
    with st.chat_message("assistant"):
        st.caption(f"⏱️ 后台执行中：{running_task.get('stage', '处理中')}")
        live_tools = running_task.get("tool_calls") or []
        if live_tools:
            st.caption("🛠️ 本轮工具: " + " | ".join(live_tools))
        live_text = running_task.get("full_response", "")
        if live_text:
            st.markdown(live_text + "▌")
        else:
            st.markdown("正在后台生成回答，你可以切换到其他会话，当前任务会继续执行。")

    # 当前会话有后台任务时，自动轮询刷新页面。
    # 优先使用 Streamlit 内置 autorefresh，减少高频手动 rerun 导致的前端节点异常。
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=UI_POLL_INTERVAL_MS, key=f"task_poll_{st.session_state.active_session_id}")
    else:
        time.sleep(UI_POLL_INTERVAL)
        st.rerun()

# --- 输入框处理 ---
if prompt := st.chat_input("请输入关于集成电路的问题..."):
    exists_running = get_running_task_for_session(st.session_state.active_session_id)
    if exists_running:
        st.warning("当前会话已有任务在后台运行，请等待完成后再发送新问题。")
    else:
        # 1. 写入用户问题并持久化
        st.session_state.messages.append({"role": "user", "content": prompt})
        persist_active_session_messages()

        # 2. 启动后台任务，不阻塞当前页面
        start_generation_task(st.session_state.active_session_id, prompt)
        st.rerun()