import os
import shutil
import hashlib
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class VASMasterIndexer:
    def __init__(self, storage_path="vector_db/"):
        self.storage_path = storage_path
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Tách 5 tầng tri thức
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Standard"),
                ("##", "Chapter"),
                ("###", "Section"),
                ("####", "Article"),
                ("#####", "Point")
            ],
            strip_headers=True
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200, # Overlap để giữ tính liên kết khi phải cắt đoạn
            separators=["\n\n", "\n", ". ", " ", ""] 
        )

    def _clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def run(self, md_dir):
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)

        md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
        vector_db = None

        for file_name in md_files:
            print(f"📂 Indexing tri thức: {file_name}")
            with open(os.path.join(md_dir, file_name), "r", encoding="utf-8") as f:
                md_content = f.read()

            # Tách theo Header trước (tạo ra các tảng tri thức theo logic)
            header_chunks = self.header_splitter.split_text(md_content)
            
            final_chunks = []

            for chunk in header_chunks:
                # Kiểm tra độ dài. Nếu tảng này quá to (>2000), ta băm nhỏ nó ra thành các sub_chunks trước khi bơm ngữ cảnh.
                # Nếu nhỏ hơn 2000, sub_chunks sẽ chỉ chứa chính nó.
                sub_chunks = self.text_splitter.split_documents([chunk])

                for sub in sub_chunks:
                    # Bơm ngữ cảnh vào từng mảnh nhỏ sau khi đã cắt xong
                    m = sub.metadata
                    hierarchy = [str(v) for k, v in m.items() if v]
                    prefix = " > ".join(hierarchy)
                    
                    clean_body = self._clean_text(sub.page_content)
                    if not clean_body: continue 
                    
                    sub.page_content = f"【NGỮ CẢNH: {prefix}】\nNỘI DUNG: {clean_body}"
                    
                    sub.metadata = {k: str(v) for k, v in m.items()}
                    sub.metadata["source"] = file_name
                    
                    final_chunks.append(sub)

            # Nạp vào ChromaDB
            if final_chunks:
                ids = [hashlib.md5(f"{file_name}_{idx}_{c.page_content[:40]}".encode()).hexdigest() 
                       for idx, c in enumerate(final_chunks)]
                
                if vector_db is None:
                    vector_db = Chroma.from_documents(
                        documents=final_chunks,
                        embedding=self.embeddings,
                        persist_directory=self.storage_path,
                        collection_name="vas_expert_db",
                        ids=ids
                    )
                else:
                    vector_db.add_documents(documents=final_chunks, ids=ids)
                print(f"Đã nạp {len(final_chunks)} chunks chuẩn hóa.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    md_dir = os.path.join(project_root, "processed")
    db_path = os.path.join(project_root, "vector_db")
    
    indexer = VASMasterIndexer(storage_path=db_path)
    indexer.run(md_dir)