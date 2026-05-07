import pandas as pd
import json
import time
import os
import re
import ast
import gc
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from dotenv import load_dotenv
from core.hybrid_engine import VASHybridSystem 
from core.local_engine import VASLocalSystem 
from core.cloud_engine import VASCloudSystem 

load_dotenv()

class RAGEvaluator:
    def __init__(self, excel_input, db_path, output_excel=None):
        self.excel_input = excel_input
        self.db_path = db_path
        if output_excel is None:
            self.output_excel = os.path.join(project_root, "evaluation_results", "Evaluation_Final_Results.xlsx")
        else:
            self.output_excel = output_excel
        self.temp_csv = "eval_results.csv"
        
        print("\nĐang khởi tạo các engine (Local, Hybrid, Cloud)...")
        self.systems = {
            "local": VASLocalSystem(db_path),
            "hybrid": VASHybridSystem(db_path),
            "cloud": VASCloudSystem(db_path)
        }

    def format_metadata_to_path(self, meta):
        if not meta: return "None"
        ordered_keys = ['Standard', 'Chapter', 'Section', 'Article', 'Point']
        path_parts = [str(meta.get(k)) for k in ordered_keys if meta.get(k)]
        return " ➔ ".join(path_parts) if path_parts else "N/A"

    def parse_history(self, history_cell):
        if pd.isna(history_cell) or str(history_cell).strip() == "": return []
        try: return json.loads(history_cell)
        except:
            try: return ast.literal_eval(history_cell)
            except: return []

    def save_to_csv(self, row_dict):
        file_exists = os.path.isfile(self.temp_csv)
        with open(self.temp_csv, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def run(self):
        if not os.path.exists(self.excel_input):
            print(f"Lỗi: Không thấy file {self.excel_input}")
            return

        df_input = pd.read_excel(self.excel_input)
        total_rows = len(df_input)

        if os.path.exists(self.temp_csv): os.remove(self.temp_csv)

        print(f"\nBắt đầu đánh giá {total_rows} kịch bản...")

        for index, row in df_input.iterrows():
            q_id = row['ID']
            question = row['Question']
            history = self.parse_history(row.get('History', []))
            
            print(f"\n" + "-"*50)
            print(f"🔄 [{index + 1}/{total_rows}] Đang xử lý ID: {q_id}")

            for mode in ["local", "hybrid", "cloud"]:
                print(f"{'='*10} Mode: {mode.upper()} {'='*10}", end=" ", flush=True)
                
                start_time = time.time()
                try:
                    output = self.systems[mode].run(question, history)
                    latency = round(time.time() - start_time, 2)
                    
                    # Xử lý nội dung hiển thị
                    meta_list = [f"{i+1}. {self.format_metadata_to_path(s.get('metadata', {}))}" 
                                 for i, s in enumerate(output['sources'])]
                    cont_list = [f"--- NGUỒN {i+1} ---\n{s.get('content', '').strip()}" 
                                 for i, s in enumerate(output['sources'])]
                    
                    res_row = row.to_dict()
                    res_row.update({
                        "mode": mode,
                        "standalone_query": output.get('standalone_query', ""),
                        "keywords": ", ".join(output.get('keywords', [])) if isinstance(output.get('keywords'), list) else "",
                        "retrieved_metadata": "\n".join(meta_list),
                        "retrieved_contexts": "\n\n".join(cont_list),
                        "answer": output.get('answer', ""),
                        "latency": latency
                    })
                    
                    self.save_to_csv(res_row)
                    print(f"   Chế độ {mode.upper()} chạy xong với ({latency}s)")

                    if mode in ["hybrid", "cloud"]:
                        time.sleep(3) # Nghỉ nhẹ để tránh Rate Limit Cloud

                except Exception as e:
                    print(f"   Lỗi! {str(e)}")
                    res_row = row.to_dict()
                    res_row.update({"mode": mode, "answer": f"LỖI: {str(e)}", "latency": 0})
                    self.save_to_csv(res_row)

            gc.collect()

        if os.path.exists(self.temp_csv):
            print("\nĐang chuyển đổi dữ liệu sang Excel...")
            final_df = pd.read_csv(self.temp_csv)
            final_df.to_excel(self.output_excel, index=False)
            print(f"HOÀN THÀNH! Kết quả tại: {self.output_excel}")

if __name__ == "__main__":
    evaluator = RAGEvaluator(os.path.join(project_root, "data", "test_dataset.xlsx"), os.path.join(project_root, "vas_vector_db"))
    evaluator.run()
