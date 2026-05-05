import streamlit as st
import os
from rag_engine import VASExpertSystem
from local_rag_engine import VASLocalSystem

st.set_page_config(page_title="VAS Expert RAG", page_icon="📑", layout="wide")

# API Key cho Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyCE6ej48wj5AOBkztExNrsgq8e0zJMWjdo" 

with st.sidebar:
    st.header("⚙️ Cấu hình")
    mode = st.radio("Chế độ:", ("Model 1: Local (1 model, no NLI)", "Model 2: Hybrid Expert"), index=1)
    is_hybrid = "Model 2" in mode
    if st.button("Làm mới chat"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_bot(hybrid):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), "vector_db")
    return VASExpertSystem(db_path) if hybrid else VASLocalSystem(db_path)

bot = load_bot(is_hybrid)

st.title("📑 Hệ thống Trợ lý Chuẩn mực Kế toán (VAS RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Hỏi về VAS ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Hệ thống đang xử lý...", expanded=False):
            result = bot.run(prompt, st.session_state.messages[:-1])
            
            st.write(f"**Câu hỏi gốc:** {result['original_query']}")
            st.write(f"**Truy vấn độc lập:** {result['standalone_query']}")
            st.write(f"**Từ khóa:** {', '.join(result['keywords'])}")
            st.divider()
            
            st.write("### 📚 Cơ sở tri thức tìm thấy:")
            for i, src in enumerate(result["sources"]):
                meta = src['metadata']

                ordered_keys = ['Standard', 'Chapter', 'Section', 'Article', 'Point']
                path = " ➔ ".join([str(meta.get(k)) for k in ordered_keys if meta.get(k)])
                
                # set expanded=True để user thấy luôn
                with st.expander(f"📍 Nguồn {i+1}: {path}", expanded=True):
                    raw_content = src['content']
                    if "NỘI DUNG:" in raw_content:
                        # Lấy phần văn bản sau chữ "NỘI DUNG:"
                        clean_content = raw_content.split("NỘI DUNG:")[-1].strip()
                    else:
                        clean_content = raw_content
                        
                    st.info(clean_content)
        
        # Hiển thị câu trả lời cuối cùng
        st.markdown(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})