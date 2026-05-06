import json
import os
from datetime import datetime

class ChatManager:
    def __init__(self, storage_dir="history"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def save_chat(self, session_id, messages, mode):
        if not messages: return
        # Giới hạn tiêu đề ngắn lại để không làm vỡ giao diện sidebar
        raw_title = messages[0]['content']
        title = (raw_title[:20] + '...') if len(raw_title) > 20 else raw_title
        
        data = {
            "session_id": session_id,
            "title": title,
            "mode": mode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages
        }
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def delete_chat(self, session_id):
        """Xóa file lịch sử chat"""
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def list_chats(self):
        chats = []
        if not os.path.exists(self.storage_dir): return chats
        for file in os.listdir(self.storage_dir):
            if file.endswith(".json"):
                path = os.path.join(self.storage_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        meta = json.load(f)
                        chats.append({
                            "id": meta['session_id'],
                            "title": meta.get('title', 'Chat'),
                            "timestamp": meta.get('timestamp', '')
                        })
                    except: continue
        return sorted(chats, key=lambda x: x['timestamp'], reverse=True)

    def load_chat(self, session_id):
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None