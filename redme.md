# IC-Expert Agent Infra

集成电路领域 ReAct + RAG Agent，支持两种启动模式：

1. 本地开发模式（默认云端 LLM）
2. Docker 部署模式（默认本地 Ollama）

## 核心能力

- IC 知识检索工具：ic_rag_search
- Verilog 静态审查工具：verilog_code_analyzer
- 时序约束建议工具：timing_constraint_suggester
- 严格模式：检索未命中时固定拒答
- 参考资料可追责：服务端强制写入真实检索来源

## 配置分层

项目已按环境分层，避免本地与 Docker 混用变量：

- .env.dev：本地开发配置（默认云端 LLM）
- .env.docker：Docker 配置（默认本地 Ollama）
- .env.prod：生产配置模板

程序通过 APP_ENV 自动选择配置：

- APP_ENV=dev -> .env.dev
- APP_ENV=docker -> .env.docker
- APP_ENV=prod -> .env.prod

## 一键命令

项目提供 Makefile 作为统一入口：

```bash
make help
```

常用命令：

```bash
make install      # 安装依赖
make run-local    # 本地一键启动（API + UI）
make run-api      # 仅启动后端
make run-ui       # 仅启动前端（streamlit run app.py）
make docker-up    # Docker 一键启动
make docker-down  # Docker 停止
make docker-logs  # 查看 Docker 日志
```

## 启动方式说明

### 方式一：本地开发（云端 LLM）

```bash
make run-local
```

默认读取 .env.dev，适合调试与迭代。

访问地址：

- FastAPI: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501

### 方式二：Docker 部署（本地 Ollama）

```bash
make docker-up
```

默认读取 .env.docker，适合本地部署验证。

访问地址：

- FastAPI: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501

## 评测

评测脚本：evaluate_ragas.py

```bash
python evaluate_ragas.py
```

输出文件：

- ragas_evaluation_report.csv（总指标）
- ragas_evaluation_details.csv（逐条明细，含 citation_correctness / refusal_correctness）
