import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def export_db_to_excel(storage_path=None):
    if storage_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        storage_path = os.path.join(project_root, "vas_vector_db")
    print("Đang kết nối tới Database để trích xuất dữ liệu...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if not os.path.exists(storage_path):
        print(f"Không tìm thấy thư mục database tại: {storage_path}")
        return

    db = Chroma(
        persist_directory=storage_path, 
        embedding_function=embeddings, 
        collection_name="vas_expert_db"
    )
    
    data = db.get(include=['metadatas', 'documents'])
    
    if not data['documents']:
        print("Database hiện tại đang trống.")
        return

    df_meta = pd.DataFrame(data['metadatas'])
    
    # Thêm nội dung văn bản vào cột cuối
    df_meta['Nội dung Chunk'] = data['documents']
    df_meta['Độ dài (ký tự)'] = df_meta['Nội dung Chunk'].apply(len)

    # Ưu tiên các cột quan trọng lên trước
    priority_cols = ['Document', 'Chapter', 'Section', 'Article', 'Point', 'source']
    existing_priority = [c for c in priority_cols if c in df_meta.columns]
    other_cols = [c for c in df_meta.columns if c not in existing_priority and c != 'Nội dung Chunk']
    
    final_order = existing_priority + other_cols + ['Nội dung Chunk']
    df_final = df_meta[final_order]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_name = os.path.join(project_root, "data", "Kiem_tra_tri_thuc_VAS.xlsx")
    df_final.to_excel(file_name, index=False, engine='openpyxl')
    
    print("-" * 50)
    print(f"THÀNH CÔNG! Đã xuất {len(df_final)} chunks ra file Excel.")
    print(f"Đường dẫn file: {os.path.abspath(file_name)}")
    print("-" * 50)

if __name__ == "__main__":
    export_db_to_excel()
