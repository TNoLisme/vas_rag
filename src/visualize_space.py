import pandas as pd
import os
from renumics import spotlight
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def visualize_rag_space():
    # 1. Cấu hình đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "vector_db")
    
    print(f"🚀 Đang nạp dữ liệu từ: {db_path}...")
    
    # 2. Khởi tạo Embeddings và kết nối DB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings, 
        collection_name="vas_expert_db"
    )
    
    # 3. Lấy dữ liệu (Bắt buộc lấy cả 'embeddings')
    data = db.get(include=['embeddings', 'metadatas', 'documents'])
    
    if data['embeddings'] is None or len(data['embeddings']) == 0:
        print("❌ Database rỗng hoặc không chứa embeddings!")
        return

    # 4. Tạo DataFrame để Spotlight đọc
    df = pd.DataFrame(data['metadatas'])
    df['text'] = data['documents']
    df['embedding'] = list(data['embeddings']) # Đây là cột quan trọng để vẽ bản đồ
    
    # Tạo cột nhãn gộp để dễ quan sát trên bản đồ
    df['full_path'] = df.apply(lambda r: f"{r.get('Chapter', '')} > {r.get('Article', '')}", axis=1)

    print("🌐 Đang khởi chạy Spotlight tại http://localhost:8000 ...")
    # 5. Hiển thị giao diện
    spotlight.show(df)

if __name__ == "__main__":
    visualize_rag_space()