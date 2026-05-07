import os
import re
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class VASHybridSystem:
    def __init__(self, vector_db_path="vas_vector_db/"):
        print("\n" + "=" * 60)
        print("[INIT] HYBRID ADAPTIVE SELF-RAG")
        
        # Models
        self.local_llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
        self.cloud_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
        
        # Database & Embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings,
            collection_name="vas_expert_db"
        )

        # Lấy dữ liệu để khởi tạo BM25
        all_data = self.vector_db.get()
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data['documents'], all_data['metadatas'])
        ]

        # Khởi tạo 2 bộ máy tìm kiếm riêng biệt
        self.vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5 # Lấy 5 đoạn chứa từ khóa chính xác nhất

        print(f"[INIT] Đã nạp {len(documents)} chunks tri thức.")
        print("-" * 60 + "\n")

    # NODE REWRITE
    def node_rewrite(self, user_query, history_str):
        print(f"[NODE: REWRITE] Đang phân tích câu hỏi...")
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
        
        response = self.local_llm.invoke(prompt.format(history=history_str, query=user_query)).content
        try:
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            pkg = json.loads(json_str)
            return pkg.get('standalone_query', user_query), pkg.get('keywords', [])
        except:
            return user_query, []

    def node_retrieve(self, standalone, keywords):
        print(f"[NODE: RETRIEVE] Thực hiện Hybrid Search...")
        
        vec_docs = self.vector_retriever.invoke(standalone) # Trả về k=5
        kw_str = " ".join(keywords)
        bm25_docs = self.bm25_retriever.invoke(kw_str)        # Trả về k=3
        
        # Thuật toán Reranking/Fusion đơn giản
        # Dùng một dictionary để lưu trữ: content_hash -> {doc_object, score}
        ranked_results = {}
        
        # Trọng số ưu tiên
        vector_weight = 1.0
        bm25_weight = 0.8 # Thường Vector Search được tin cậy hơn một chút về ngữ cảnh
        
        # Duyệt qua kết quả Vector
        for i, doc in enumerate(vec_docs):
            content = doc.page_content
            score = vector_weight * (1.0 / (i + 1)) # Ví dụ: Top 1 = 1.0đ, Top 2 = 0.5đ
            ranked_results[content] = {"doc": doc, "score": score}
            
        # Duyệt qua kết quả BM25
        for i, doc in enumerate(bm25_docs):
            content = doc.page_content
            score = bm25_weight * (1.0 / (i + 1))
            
            if content in ranked_results:
                # NẾU TRÙNG LẶP: Cộng dồn điểm
                ranked_results[content]["score"] += score
            else:
                ranked_results[content] = {"doc": doc, "score": score}
        
        # Sắp xếp lại toàn bộ theo điểm số từ cao xuống thấp
        sorted_results = sorted(ranked_results.values(), key=lambda x: x["score"], reverse=True)
        
        # Lấy lại đối tượng Document đã được sắp xếp
        final_docs = [item["doc"] for item in sorted_results]
        
        # Giới hạn lấy lại Top 5 hoặc Top 7 mảnh tốt nhất sau khi đã trộn
        return final_docs[:6]

    # NODE SUFFICIENCY
    def node_check_sufficiency(self, user_query, docs):
        print(f"[NODE: SUFFICIENCY CHECK] Đang đánh giá tiềm năng tài liệu...")
        
        # Lấy context rộng hơn một chút để AI có dữ liệu bao quát
        context_text = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""[BỐI CẢNH] Bạn là trợ lý phân tích dữ liệu cho hệ thống RAG kế toán.
[NHIỆM VỤ] Đánh giá xem TÀI LIỆU có cung cấp ĐỦ CƠ SỞ để AI suy luận và trả lời CÂU HỎI hay không.
[TÀI LIỆU]
{context_text}
[CÂU HỎI]
{user_query}
[TIÊU CHÍ ĐÁNH GIÁ]
- Trả về YES: Nếu tài liệu nói trực tiếp về chủ đề của câu hỏi và chứa các dữ liệu, quy định hoặc khái niệm liên quan mà từ đó AI có thể tổng hợp thành câu trả lời hoàn chỉnh.
- Trả về NO: Chỉ khi tài liệu hoàn toàn KHÔNG liên quan, nội dung trọng tâm không nói về nội dung câu hỏi cần trả lời, nói về một chủ đề khác, hoặc không chứa bất kỳ từ khóa/nguyên tắc nào liên quan đến câu hỏi.
Lưu ý: Bạn không cần đáp án hoàn chỉnh, chỉ cần có đủ "nguyên liệu" liên quan đến nội dung của câu hỏi để suy luận.
CHỈ TRẢ VỀ: YES hoặc NO.
[KẾT QUẢ]:"""

        response = self.local_llm.invoke(prompt).content.strip()
        decision = "YES" if "YES" in response.upper() else "NO"
        return decision

    def node_generate(self, model, user_query, docs):
        model_name = "CLOUD GEMINI" if "Google" in str(type(model)) else "LOCAL QWEN"
        print(f"[NODE: GENERATE] Model: {model_name} | Đang soạn thảo...")
        
        # Chuẩn bị ngữ cảnh với đường dẫn phân cấp rõ ràng
        context_text = ""
        for i, d in enumerate(docs):
            # Tạo nhãn nguồn chuyên nghiệp
            path = " > ".join([str(v) for k, v in d.metadata.items() if k in ['Standard', 'Chapter', 'Section', 'Article', 'Point'] and v])
            context_text += f"--- DỮ LIỆU {i+1} (Nguồn: {path}) ---\n{d.page_content}\n\n"

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

        answer = model.invoke(prompt).content
        return answer

    # NODE NLI
    def node_verify_nli(self, answer, docs):
        print(f"[NODE: NLI CHECK] Đang kiểm tra độ trung thực (Anti-Hallucination)...")
        
        context_text = "\n\n".join([d.page_content for d in docs])[:3500]
        
        # Prompt này ép Qwen 3B đóng vai một "Kiểm toán viên" bắt lỗi
        prompt = f"""[VAI TRÒ] Bạn là Kiểm toán viên nội bộ chuyên bắt lỗi ảo giác thông tin.
[NHIỆM VỤ] So sánh CÂU TRẢ LỜI với TÀI LIỆU GỐC để xác định tính trung thực.
[TÀI LIỆU GỐC]
{context_text}
[CÂU TRẢ LỜI CỦA AI]
{answer}
[QUY TẮC KIỂM TRA]
1. Trả về YES: Nếu mọi thông tin trong CÂU TRẢ LỜI (bao gồm cả các suy luận) đều có CƠ SỞ từ TÀI LIỆU GỐC.
2. Trả về NO: Nếu CÂU TRẢ LỜI chứa các con số, mã hiệu, hoặc quy định KHÔNG hề có trong TÀI LIỆU GỐC.
CHỈ TRẢ VỀ DUY NHẤT: YES hoặc NO.
[KẾT QUẢ]:"""

        # Sử dụng Local LLM để tiết kiệm chi phí
        res = self.local_llm.invoke(prompt).content.strip()
        decision = "YES" if "YES" in res.upper() else "NO"
        return decision

    def run(self, user_query, chat_history_list):
        print(f"\n[TRUY VẤN HYBRID]: {user_query}")
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history_list[-3:]])
        
        # REWRITE
        standalone, keywords = self.node_rewrite(user_query, history_str)
        print(f"   ➔ Câu hỏi độc lập: {standalone}")
        print(f"   ➔ Keywords lọc được: {keywords}")

        # ADAPTIVE RETRIEVAL LOOP
        attempts = 0
        final_docs = []
        is_sufficient = False
        current_standalone = standalone

        while attempts < 3:
            final_docs = self.node_retrieve(current_standalone, keywords)
            
            if self.node_check_sufficiency(user_query, final_docs):
                print("   Kết quả: Đủ thông tin.")
                is_sufficient = True
                break
            else:
                print(f"   Lần {attempts+1} thất bại. Đang viết lại query tìm kiếm...")
                refine_prompt = f"""Bạn là chuyên gia tra cứu văn bản pháp luật. 
Lần tìm kiếm trước cho câu hỏi "{user_query}" không mang lại đủ thông tin.

[NHIỆM VỤ] Hãy viết lại một câu truy vấn tìm kiếm mới:
- Sử dụng các thuật ngữ kế toán đồng nghĩa (Ví dụ: "ghi nhận" thay cho "tính", "định khoản" thay cho "hạch toán").
- Nếu có mã chuẩn mực (VAS) hoặc thông tư, hãy giữ nguyên.
- Mục tiêu: Tìm được đúng đoạn văn bản chứa quy định chi tiết.

[TRẢ VỀ]: Duy nhất câu truy vấn mới, không giải thích."""

                current_standalone = self.local_llm.invoke(refine_prompt).content.strip()
                attempts += 1

        # ESCALATION & GENERATION
        if not is_sufficient:
            print("[ESCALATION] Leo thang lên Gemini Cloud (k=10)...")
            final_docs = self.vector_db.similarity_search(standalone, k=10)
            model = self.cloud_llm
        else:
            model = self.local_llm

        answer = self.node_generate(model, user_query, final_docs)

        # FINAL NLI CHECK
        is_faithful = self.node_verify_nli(answer, final_docs)
        print(f"   Kết quả NLI: {'TRUNG THỰC' if is_faithful else 'ẢO GIÁC'}")

        if not is_faithful:
            print("   Phát hiện ảo giác! Nhờ Cloud Gemini sửa lỗi...")
            # Chuẩn bị context để gửi kèm cho Gemini (rất quan trọng)
            context_text = "\n\n".join([d.page_content for d in final_docs])
    
            correction_prompt = f"""[CẢNH BÁO] Câu trả lời trước đó đã vi phạm kiểm tra tính trung thực (NLI) và chứa thông tin ảo giác không có trong tài liệu.
[TRI THỨC GỐC - NGUỒN DUY NHẤT]
{context_text}
[CÂU HỎI]
{user_query}
[CÂU TRẢ LỜI LỖI (CẦN SỬA)]
{answer}
[YÊU CẦU NGHIÊM NGẶT]
1. Hãy viết lại câu trả lời hoàn toàn mới, CHỈ dựa trên [TRI THỨC GỐC].
2. Loại bỏ tất cả thông tin không có bằng chứng trong tài liệu.
3. Trích dẫn lại chính xác Điều/Khoản hoặc mã Nguồn.
4. Nếu tài liệu vẫn không đủ ý, hãy thừa nhận "Tài liệu hiện tại không đề cập đến..." thay vì suy diễn.
CÂU TRẢ LỜI ĐÃ VIẾT LẠI:"""

            answer = self.cloud_llm.invoke(correction_prompt).content

        # Chuyển đổi thành định dạng cho UI
        sources_dict = [{"content": d.page_content, "metadata": d.metadata} for d in final_docs]

        return {
            "original_query": user_query,
            "standalone_query": standalone,
            "keywords": keywords,
            "answer": answer,
            "sources": sources_dict
        }
