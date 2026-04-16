import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载 .env 里的 API Key
load_dotenv()

# 2. 初始化大模型 (这个 "chef" 大厨)
# model_name 可以根据你用的平台改，比如 deepseek-chat 或 moonshot-v1-8k
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "qwen-plus"),
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

# 3. 测试调用
print("正在连接大模型...")
try:
    response = llm.invoke("你好，请用一句话介绍什么是集成电路？")
    print("--------------------------------------------------")
    print("模型回答：")
    print(response.content)
    print("--------------------------------------------------")
    print("✅ 成功！你的环境配置完美！")
except Exception as e:
    print("❌ 失败！请检查你的 API Key 或网络。")
    print(e)