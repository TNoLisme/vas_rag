import ast
import argparse
import csv
import gc
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.cloud_engine import VASCloudSystem
from core.hybrid_engine import VASHybridSystem
from core.local_engine import VASLocalSystem

load_dotenv()


class RAGResponseRunner:
    MODES = ["local", "hybrid", "cloud"]

    REQUIRED_COLUMNS = [
        "ID",
        "Type",
        "Question",
        "History",
        "Ground_Truth_Query",
        "Ground_Truth_Metadata",
        "Ground_Truth_Answer",
    ]
    RESULT_COLUMNS = [
        "mode",
        "standalone_query",
        "keywords",
        "retrieved_metadata",
        "retrieved_contexts",
        "answer",
        "latency",
        "error",
    ]

    def __init__(self, excel_input, db_path, output_excel=None, rate_limit_sleep=3, modes=None):
        self.excel_input = excel_input
        self.db_path = db_path
        self.rate_limit_sleep = rate_limit_sleep
        self.modes = modes or self.MODES
        invalid_modes = sorted(set(self.modes) - set(self.MODES))
        if invalid_modes:
            raise ValueError(f"Mode không hợp lệ: {invalid_modes}. Chọn trong {self.MODES}")
        self.output_excel = output_excel or os.path.join(
            project_root, "evaluation_results", "Evaluation_Final_Results.xlsx"
        )
        self.output_dir = os.path.dirname(self.output_excel)
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_csv = os.path.join(self.output_dir, "eval_results.csv")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Không tìm thấy GOOGLE_API_KEY trong môi trường hoặc file .env.")
        os.environ["GOOGLE_API_KEY"] = api_key

        print("\nĐang khởi tạo các engine...")
        system_factories = {
            "local": VASLocalSystem,
            "hybrid": VASHybridSystem,
            "cloud": VASCloudSystem,
        }
        self.systems = {mode: system_factories[mode](db_path) for mode in self.modes}

    def format_metadata_to_path(self, meta):
        if not meta:
            return "None"
        keys = ["Standard", "Chapter", "Section", "Article", "Point"]
        path_parts = [str(meta.get(k)) for k in keys if meta.get(k)]
        return " ➔ ".join(path_parts) if path_parts else "N/A"

    def parse_history(self, history_cell):
        if pd.isna(history_cell) or str(history_cell).strip() == "":
            return []
        try:
            history = json.loads(history_cell)
        except Exception:
            history = ast.literal_eval(history_cell)
        return history

    def validate_input(self, df_input):
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df_input.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc trong test dataset: {missing}")

    def format_sources(self, sources):
        retrieved_metadata = []
        retrieved_contexts = []
        for source in sources:
            metadata = source.get("metadata", {}) or {}
            content = str(source.get("content", "")).strip()
            retrieved_metadata.append(
                {
                    "path": self.format_metadata_to_path(metadata),
                    "metadata": metadata,
                }
            )
            retrieved_contexts.append(content)
        return (
            json.dumps(retrieved_metadata, ensure_ascii=False),
            json.dumps(retrieved_contexts, ensure_ascii=False),
        )

    def save_to_csv(self, row_dict, fieldnames):
        file_exists = os.path.isfile(self.temp_csv)
        with open(self.temp_csv, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({col: row_dict.get(col, "") for col in fieldnames})

    def run(self, limit=None, save_excel=True, save_csv=True):
        if not os.path.exists(self.excel_input):
            raise FileNotFoundError(f"Không thấy file test dataset: {self.excel_input}")

        df_input = pd.read_excel(self.excel_input).fillna("")
        self.validate_input(df_input)
        if limit is not None:
            df_input = df_input.head(limit)
        total_rows = len(df_input)
        output_columns = list(df_input.columns) + self.RESULT_COLUMNS

        if save_csv and os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)

        print(f"\nBắt đầu thu thập phản hồi cho {total_rows} kịch bản...")
        results = []

        for index, row in df_input.iterrows():
            q_id = row["ID"]
            question = row["Question"]
            history = self.parse_history(row.get("History", ""))

            print("\n" + "-" * 50)
            print(f"[{index + 1}/{total_rows}] Đang xử lý ID: {q_id}")

            for mode in self.modes:
                print(f"{'=' * 10} Mode: {mode.upper()} {'=' * 10}", end=" ", flush=True)

                start_time = time.time()
                res_row = row.to_dict()
                try:
                    output = self.systems[mode].run(question, history)
                    latency = round(time.time() - start_time, 2)

                    retrieved_metadata, retrieved_contexts = self.format_sources(
                        output.get("sources", [])
                    )

                    res_row.update(
                        {
                            "mode": mode,
                            "standalone_query": output.get("standalone_query", ""),
                            "keywords": ", ".join(output.get("keywords", []))
                            if isinstance(output.get("keywords"), list)
                            else "",
                            "retrieved_metadata": retrieved_metadata,
                            "retrieved_contexts": retrieved_contexts,
                            "answer": output.get("answer", ""),
                            "latency": latency,
                            "error": "",
                        }
                    )
                    print(f"xong ({latency}s)")

                    if mode in ["hybrid", "cloud"] and self.rate_limit_sleep > 0:
                        time.sleep(self.rate_limit_sleep)

                except Exception as e:
                    latency = round(time.time() - start_time, 2)
                    res_row.update(
                        {
                            "mode": mode,
                            "answer": f"LỖI: {e}",
                            "latency": latency,
                            "error": str(e),
                        }
                    )
                    print(f"lỗi: {e}")

                results.append(res_row)
                if save_csv:
                    self.save_to_csv(res_row, output_columns)

            gc.collect()

        final_df = pd.DataFrame(results, columns=output_columns)
        if save_excel:
            print("\nĐang lưu kết quả Excel...")
            final_df.to_excel(self.output_excel, index=False)
            print(f"HOÀN THÀNH. Kết quả tại: {self.output_excel}")
        return final_df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAG responses for evaluation.")
    parser.add_argument("--input", default=os.path.join(project_root, "data", "test_dataset.xlsx"))
    parser.add_argument("--db", default=os.path.join(project_root, "vas_vector_db"))
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "evaluation_results", "Evaluation_Final_Results.xlsx"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate-limit-sleep", type=float, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RAGResponseRunner.MODES,
        default=RAGResponseRunner.MODES,
    )
    parser.add_argument("--no-save-excel", action="store_true")
    parser.add_argument("--no-save-csv", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = RAGResponseRunner(
        args.input,
        args.db,
        output_excel=args.output,
        rate_limit_sleep=args.rate_limit_sleep,
        modes=args.modes,
    )
    runner.run(
        limit=args.limit,
        save_excel=not args.no_save_excel,
        save_csv=not args.no_save_csv,
    )
