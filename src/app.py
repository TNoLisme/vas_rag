import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from rag_engine import VASHybridSystem
from local_rag_engine import VASLocalSystem
from cloud_rag_engine import VASCloudSystem
from chat_manager import ChatManager

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Không tìm thấy GOOGLE_API_KEY.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(page_title="VAS Expert RAG", page_icon="📑", layout="wide")

chat_manager = ChatManager()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Model 2: Hybrid Expert"

with st.sidebar:
    st.header("📑 VAS RAG")
    
    if st.button("➕ Tạo cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("⚙️ Cấu hình")
    mode = st.radio(
        "Mode:",
        ("Local Only", "Hybrid Expert", "Cloud Only"),
        index=1 if st.session_state.current_mode == "Hybrid Expert" else (0 if st.session_state.current_mode == "Local Only" else 2)
    )
    st.session_state.current_mode = mode

    st.divider()
    st.subheader("🕒 Lịch sử trò chuyện")
    
    past_chats = chat_manager.list_chats()
    for chat in past_chats:
        with st.container():
            col_main, col_del = st.columns([0.8, 0.2])
            
            with col_main:
                if st.button(f"💬 {chat['title']}", key=f"msg_{chat['id']}", use_container_width=True):
                    loaded_data = chat_manager.load_chat(chat['id'])
                    if loaded_data:
                        st.session_state.messages = loaded_data['messages']
                        st.session_id = loaded_data['session_id']
                        st.session_state.current_mode = loaded_data['mode']
                        st.rerun()
            
            with col_del:
                if st.button("❌", key=f"del_{chat['id']}", help="Xóa cuộc trò chuyện này"):
                    chat_manager.delete_chat(chat['id'])
                    if st.session_state.session_id == chat['id']:
                        st.session_state.messages = []
                        st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()

@st.cache_resource
def load_bot(selected_mode):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), "vector_db")
    if selected_mode == "Hybrid Expert":
        return VASHybridSystem(db_path)
    elif selected_mode == "Local Only":
        return VASLocalSystem(db_path)
    else:
        return VASCloudSystem(db_path)

bot = load_bot(st.session_state.current_mode)

st.title(f"📑 Trợ lý Kế toán (VAS)")
st.caption(f"Chế độ hiện tại: {st.session_state.current_mode}")

# Hiển thị tin nhắn cũ
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Nhập tin nhắn mới
if prompt := st.chat_input("Hỏi về VAS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Hệ thống đang xử lý...", expanded=False):
            result = bot.run(prompt, st.session_state.messages[:-1])
            
            st.write(f"**Câu hỏi gốc:** {result.get('original_query', prompt)}")
            st.write(f"**Truy vấn độc lập:** {result.get('standalone_query', '')}")
            st.write(f"**Từ khóa:** {', '.join(result.get('keywords', []))}")
            st.divider()
            
            st.write("### Cơ sở tri thức tìm thấy:")
            for i, src in enumerate(result["sources"]):
                meta = src['metadata']
                ordered_keys = ['Standard', 'Chapter', 'Section', 'Article', 'Point']
                path = " ➔ ".join([str(meta.get(k)) for k in ordered_keys if meta.get(k)])
                
                with st.expander(f"📍 Nguồn {i+1}: {path}", expanded=True):
                    raw_content = src['content']
                    clean_content = raw_content.split("NỘI DUNG:")[-1].strip() if "NỘI DUNG:" in raw_content else raw_content
                    st.info(clean_content)
        
        st.markdown(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        
        # Lưu chat vào JSON
        chat_manager.save_chat(st.session_state.session_id, st.session_state.messages, st.session_state.current_mode)
        
        # Nếu là tin nhắn đầu tiên để hiện tiêu đề bên sidebar ngay lập tức
        if len(st.session_state.messages) <= 2:
            st.rerun()