import os
import re
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from core.prompts import REWRITE_PROMPT, build_context_text, build_generation_prompt

class VASLocalSystem:
    def __init__(self, vector_db_path="vas_vector_db/"):
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
        res = self.local_llm.invoke(REWRITE_PROMPT.format(history=history_str, query=user_query)).content
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
        context_text = build_context_text(docs)
        prompt = build_generation_prompt(context_text, user_query)

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
