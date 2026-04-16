import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rag_core import get_rag_response, embeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
load_dotenv()

judge_llm = ChatOpenAI(
    model='qwen3.6-plus',  # 评测用更强的模型作为 judge
    temperature=0.0,  # 评测时保持稳定输出
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
# 在线模型评测容易受网络抖动影响；适当放宽超时并控制并发。
run_config = RunConfig(timeout=300, max_workers=1)
# ====================== IC领域测试集（你可以后续自己扩充） ======================
TEST_CASES_PATH = "eval_dataset_qg_30_pure.json"


def load_test_cases(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到测试集文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    if not isinstance(test_cases, list):
        raise ValueError("测试集格式错误：应为列表(list)")
    return test_cases


def run_ragas_evaluation():
    test_cases = load_test_cases(TEST_CASES_PATH)
    print("🔄 正在生成答案并评估（使用百炼兼容模型作为 judge）...")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    for case in test_cases:
        question = case["question"]
        ground_truth = case["ground_truth"]
        
        answer, sources = get_rag_response(question)
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append([doc.page_content for doc in sources])
        data["ground_truth"].append(ground_truth)
        print(f"✅ 已处理: {question[:60]}...")
    
    dataset = Dataset.from_dict(data)
    
    # 逐个指标评估 + 错误捕获（防止全部 NaN）
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    results = {}
    
    for metric in metrics:
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        try:
            print(f"正在计算 {metric_name} ...")
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,  # 防止单个指标卡死
            )
            raw_scores = result[metric_name]
            # ragas 0.2.x 常返回 list，统一转成数值并忽略 None/NaN 后求均值。
            numeric_scores = pd.to_numeric(pd.Series(raw_scores), errors="coerce").dropna()
            score = float(numeric_scores.mean()) if not numeric_scores.empty else None
            results[metric_name] = score
            valid_count = int(numeric_scores.shape[0])
            total_count = len(raw_scores)
            if score is None:
                print(f"⚠️ {metric_name}: 无有效分数（有效样本 {valid_count}/{total_count}）")
            else:
                print(f"✅ {metric_name}: {score:.4f}（有效样本 {valid_count}/{total_count}）")
        except Exception as e:
            print(f"⚠️ {metric_name} 计算失败: {e}")
            results[metric_name] = None
    
    # 保存完整报告
    df = pd.DataFrame([results])
    df.to_csv("ragas_evaluation_report.csv", index=False)
    
    print("\n🎉 评估完成！最终平均指标：")
    for k, v in results.items():
        print(f"   {k}: {v}")
    print("📊 完整报告已保存 → ragas_evaluation_report.csv")
    return results

if __name__ == "__main__":
    run_ragas_evaluation()