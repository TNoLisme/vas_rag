import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.evaluation_prompts import build_rewrite_judge_prompt

load_dotenv()

DEFAULT_INPUT = os.path.join(project_root, "evaluation_results", "Evaluation_Final_Results.xlsx")
DEFAULT_OUTPUT = os.path.join(project_root, "evaluation_results", "RAG_Evaluation_Summary.xlsx")
DEFAULT_DATASET = os.path.join(project_root, "data", "test_dataset.xlsx")

class RagasEvaluator:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Không tìm thấy GOOGLE_API_KEY để chạy Ragas.")
        
        self.evaluator_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            ContextRelevance(name="context_relevance"),
        ]

    def parse_json_object(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except:
            return {}

    def judge_rewrite(self, row: pd.Series) -> Dict[str, Any]:
        prompt = build_rewrite_judge_prompt(
            history=row.get("History", ""),
            original_question=row.get("Question", ""),
            rewritten_query=row.get("standalone_query", ""),
        )
        try:
            response = self.evaluator_llm.invoke(prompt)
            content = getattr(response, "content", response)
            return self.parse_json_object(str(content))
        except Exception as e:
            print(f"Lỗi khi chấm điểm Rewrite: {e}")
            return {}

    def parse_contexts(self, raw_contexts: Any) -> List[str]:
        if isinstance(raw_contexts, list):
            return [str(context).strip() for context in raw_contexts if str(context).strip()]

        text = "" if pd.isna(raw_contexts) else str(raw_contexts).strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(context).strip() for context in parsed if str(context).strip()]
        except Exception:
            pass

        contexts = [c.strip() for c in text.split("--- NGUỒN") if c.strip()]
        cleaned = []
        for context in contexts:
            context = re.sub(r"^\s*\d+\s*---\n?", "", context).strip()
            if context:
                cleaned.append(context)
        return cleaned

    def valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_df = df.copy()
        if "error" in valid_df.columns:
            valid_df = valid_df[valid_df["error"].fillna("").astype(str).str.strip() == ""]
        valid_df = valid_df[
            valid_df["answer"].fillna("").astype(str).str.strip().ne("")
            & ~valid_df["answer"].fillna("").astype(str).str.startswith("LỖI:")
        ]
        return valid_df.reset_index(drop=True)

    def prepare_ragas_dataset(self, df: pd.DataFrame) -> Dataset:
        # Ragas 0.4.x yêu cầu schema single-turn: user_input, response, retrieved_contexts, reference.
        data = {
            "user_input": df["Question"].astype(str).tolist(),
            "response": df["answer"].astype(str).tolist(),
            "retrieved_contexts": [
                self.parse_contexts(ctx) for ctx in df["retrieved_contexts"].tolist()
            ],
            "reference": df["Ground_Truth_Answer"].astype(str).tolist(),
        }
        return Dataset.from_dict(data)

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.valid_rows(df.fillna(""))
        if df.empty:
            raise ValueError("Không có dòng hợp lệ để đánh giá RAGAS sau khi lọc lỗi.")
        
        print(f"\n[PHASE 1] Chấm điểm Rewrite (Custom LLM Judge)...")
        rewrite_results = []
        for _, row in df.iterrows():
            res = self.judge_rewrite(row)
            rewrite_results.append({
                "llm_pronoun_resolution": res.get("pronoun_resolution", 0),
                "llm_standalone_quality": res.get("standalone_quality", 0),
                "llm_semantic_completeness": res.get("semantic_completeness", 0),
                "rewrite_comment": res.get("comment", "")
            })
        rewrite_df = pd.DataFrame(rewrite_results)
        
        print(f"\n[PHASE 2] Chấm điểm Generation (Ragas Framework)...")
        ragas_ds = self.prepare_ragas_dataset(df)
        
        # Chạy Ragas evaluate
        # Lưu ý: Ragas có thể tốn thời gian vì gọi API LLM liên tục
        results = evaluate(
            dataset=ragas_ds,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            raise_exceptions=False,
        )
        
        ragas_df = results.to_pandas()
        ragas_df = ragas_df.drop(
            columns=["user_input", "response", "retrieved_contexts", "reference"],
            errors="ignore",
        )
        
        # Gộp tất cả lại
        final_df = pd.concat([df, rewrite_df, ragas_df], axis=1)
        
        # Tính điểm tổng kết (Mean of Ragas metrics + Rewrite score)
        metric_columns = [
            column
            for column in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "nv_context_relevance",
            ]
            if column in final_df.columns
        ]
        final_df["overall_llm_score"] = final_df[metric_columns].mean(axis=1)
        return final_df

    def run(self, input_excel: str, output_excel: str, save_excel: bool = True):
        if not os.path.exists(input_excel):
            raise FileNotFoundError(f"Không thấy file kết quả: {input_excel}")

        df = pd.read_excel(input_excel).fillna("")
        final_df = self.evaluate_dataframe(df)
        
        if save_excel:
            os.makedirs(os.path.dirname(output_excel), exist_ok=True)
            final_df.to_excel(output_excel, index=False)
            print(f"HOÀN THÀNH. Báo cáo tại: {output_excel}")
        return final_df

def main():
    parser = argparse.ArgumentParser(description="Ragas-based Evaluator")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--no-save-excel", action="store_true")
    args = parser.parse_args()

    evaluator = RagasEvaluator(model_name=args.model)
    evaluator.run(args.input, args.output, save_excel=not args.no_save_excel)

if __name__ == "__main__":
    main()
