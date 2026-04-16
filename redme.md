### 🚀 核心成果 - RAG 可观测性指标（RAGAS 自动评估）

- **Situation**：传统 RAG 在 IC 专业领域容易出现幻觉，难以量化评估。
- **Task**：构建可观测指标体系，证明系统在 Verilog、时序约束等场景下的可靠性。
- **Action**：集成 RAGAS 评估框架，使用 llama3.1:8b 作为 judge 模型，构建 8 条 IC 领域测试集，覆盖 faithfulness、answer_relevancy 等 4 大核心指标。
- **Result**：
  - faithfulness: **0.867**（幻觉率低）
  - answer_relevancy: **0.863**
  - context_recall: **0.800**
  - context_precision: **0.733**

**指标报告**：`ragas_evaluation_report.csv`（已提交）