import os
import re
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class VASLocalSystem:
    def __init__(self, vector_db_path="vector_db/"):
        print("\n" + "-" * 60)
        print("[INIT] KHỞI TẠO HỆ THỐNG LOCAL ONLY (QUEN 3B)")
        self.local_llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings, collection_name="vas_expert_db")
        
        all_data = self.vector_db.get()
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(all_data['documents'], all_data['metadatas'])]
        
        self.vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        
        self.hybrid_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.vector_retriever], weights=[0.4, 0.6])
        print(f"[INIT] Đã nạp {len(documents)} chunks.")
        print("-" * 60 + "\n")

    def run(self, user_query, chat_history_list):
        print(f"\n[TRUY VẤN LOCAL]: {user_query}")
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history_list[-3:]])
        
        print(f"[NODE: REWRITE] Model: Local Qwen 3B")
        prompt = ChatPromptTemplate.from_template("""
[VAI TRÒ] Bạn là chuyên gia phân tích truy vấn cấp cao cho hệ thống RAG Chuẩn mực Kế toán Việt Nam (VAS).
[NHIỆM VỤ] Dựa vào lịch sử chat và câu hỏi mới, hãy thực hiện hai bước suy luận:
1. Viết lại câu hỏi thành 'standalone_query': Đây là một câu văn xuôi đầy đủ, mô tả rõ ý nghĩa ngữ nghĩa để tìm kiếm (Vector Search). Nếu người dùng dùng các từ thay thế (nó, khoản này, tài khoản đó...), hãy dựa vào lịch sử để thay bằng tên thực thể kế toán cụ thể.
2. Trích xuất 'keywords': Lọc ra 3-5 cụm từ khóa quan trọng NHẤT từ 'standalone_query'. 
- YÊU CẦU: Ưu tiên là CỤM TỪ chuyên ngành hoặc MÃ HIỆU (Ví dụ: "VAS 02", "Giá gốc hàng tồn kho"). 
- HẠN CHẾ: Hạn chế lấy các từ phổ biến, không lấy từ đơn lẻ nếu nó không mang tính đặc thù, và không lấy cả câu dài làm từ khóa.
[YÊU CẦU ĐỊNH DẠNG]
- Trả về DUY NHẤT một khối JSON. 
- KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG có văn bản bên ngoài khối JSON.
- Đảm bảo các Key trong JSON là "standalone_query" và "keywords".
[VÍ DỤ]
Lịch sử: "User: VAS 02 nói về gì? | AI: VAS 02 quy định về Hàng tồn kho."
User: "Nguyên tắc xác định giá gốc của nó?"
Kết quả JSON:
{{
    "standalone_query": "Nguyên tắc xác định giá gốc hàng tồn kho theo quy định của Chuẩn mực kế toán VAS 02 như thế nào?",
    "keywords": ["VAS 02", "giá gốc hàng tồn kho", "nguyên tắc xác định"]
}}
Lịch sử: {history}
Câu hỏi mới: {query}
Kết quả JSON:
""")

        res = self.local_llm.invoke(prompt.format(history=history_str, query=user_query)).content
        try:
            json_str = re.search(r'\{.*\}', res, re.DOTALL).group()
            pkg = json.loads(json_str)
            standalone = pkg.get('standalone_query', user_query)
            keywords = pkg.get('keywords', [])
        except:
            standalone, keywords = user_query, []

        print(f"   ➔ Standalone: {standalone}")
        print(f"   ➔ Keywords: {keywords}")

        print(f"[NODE: RETRIEVE] Model: Local Qwen 3B")
        docs = self.hybrid_retriever.invoke(f"{standalone} {' '.join(keywords)}")
        
        print(f"[NODE: GENERATE] Model: Local Qwen 3B")
        # Xây dựng context có kèm path metadata để AI dễ trích dẫn nguồn
        context_text = ""
        for i, d in enumerate(docs):
            path = " > ".join([str(v) for k, v in d.metadata.items() if k in ['Standard', 'Chapter', 'Section', 'Article', 'Point'] and v])
            context_text += f"--- NGUỒN {i+1} ({path}) ---\n{d.page_content}\n\n"

        prompt = f"""Bạn là Chuyên gia Kế toán cao cấp. Hãy trả lời câu hỏi dựa trên các nguồn tri thức được cung cấp.
[TRI THỨC CUNG CẤP]
{context_text}
[CÂU HỎI]
{user_query}
[YÊU CẦU NGHIÊM NGẶT]
1. TRUNG THỰC: Chỉ sử dụng thông tin trong [TRI THỨC CUNG CẤP]. Không dùng kiến thức ngoài.
2. SUY LUẬN LOGIC: Nếu tài liệu không có câu trả lời trực tiếp, hãy dựa vào các nguyên tắc, định nghĩa trong nguồn tin để SUY LUẬN và giải quyết câu hỏi.
3. TRÍCH DẪN (CITATION): BẮT BUỘC ghi rõ nguồn ở cuối mỗi ý hoặc mỗi đoạn. Sử dụng tên chuẩn mực hoặc số Điều/Khoản có trong nhãn (Nguồn: ...). 
   - Ví dụ: "Hàng tồn kho được tính theo giá gốc (theo VAS 02 > Điều 04)".
4. PHONG CÁCH:
- Trình bày dạng văn xuôi mạch lạc hoặc danh sách gạch đầu dòng (bullet points).
- TUYỆT ĐỐI KHÔNG sử dụng tiêu đề Markdown (không dùng dấu #, ##, ###).
- KHÔNG bôi đậm dòng để làm tiêu đề giả. Câu trả lời phải phẳng và dễ đọc như một bản ghi chú chuyên môn.
CÂU TRẢ LỜI:"""

        answer = self.local_llm.invoke(prompt).content
        print("-" * 60)

        sources_dict = [{"content": d.page_content, "metadata": d.metadata} for d in docs]

        return {
            "original_query": user_query,
            "standalone_query": standalone,
            "keywords": keywords,
            "answer": answer,
            "sources": sources_dict
        }