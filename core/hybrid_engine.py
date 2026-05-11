import os
import re
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from core.prompts import (
    REWRITE_PROMPT,
    build_context_text,
    build_generation_prompt,
    build_correction_prompt,
    build_nli_check_prompt,
    build_refine_search_query_prompt,
    build_sufficiency_check_prompt,
)

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
        response = self.local_llm.invoke(REWRITE_PROMPT.format(history=history_str, query=user_query)).content
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
        
        prompt = build_sufficiency_check_prompt(context_text, user_query)

        response = self.local_llm.invoke(prompt).content.strip()
        decision = "YES" if "YES" in response.upper() else "NO"
        return decision

    def node_generate(self, model, user_query, docs):
        model_name = "CLOUD GEMINI" if "Google" in str(type(model)) else "LOCAL QWEN"
        print(f"[NODE: GENERATE] Model: {model_name} | Đang soạn thảo...")
        
        context_text = build_context_text(docs)
        prompt = build_generation_prompt(context_text, user_query)

        answer = model.invoke(prompt).content
        return answer

    # NODE NLI
    def node_verify_nli(self, answer, docs):
        print(f"[NODE: NLI CHECK] Đang kiểm tra độ trung thực (Anti-Hallucination)...")
        
        context_text = "\n\n".join([d.page_content for d in docs])[:3500]
        
        prompt = build_nli_check_prompt(context_text, answer)

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
                refine_prompt = build_refine_search_query_prompt(user_query)

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
    
            correction_prompt = build_correction_prompt(context_text, user_query, answer)

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
