import json


def build_rewrite_judge_prompt(
    history: str,
    original_question: str,
    rewritten_query: str,
) -> str:
    payload = {
        "history": history,
        "original_question": original_question,
        "rewritten_query": rewritten_query,
    }
    return f"""Bạn là chuyên gia ngôn ngữ học đánh giá bộ tiền xử lý câu hỏi cho hệ thống RAG Kế toán.

Nhiệm vụ: Kiểm tra xem câu hỏi đã được viết lại (rewritten_query) có đầy đủ thực thể và giải quyết được các từ thay thế (nó, đó, khoản này, chuẩn mực đó...) dựa trên lịch sử chat (history) hay không.

Tiêu chí chấm điểm (0.0 đến 1.0):
1. Pronoun Resolution: Các đại từ/từ thay thế trong câu hỏi gốc đã được thay bằng tên thực thể cụ thể (ví dụ: "nó" -> "hàng tồn kho") chưa?
2. Standalone Quality: Câu hỏi mới có thể đứng độc lập mà không cần đọc lại lịch sử vẫn hiểu được nội dung tìm kiếm không?
3. Semantic Completeness: Câu hỏi mới có giữ được ý định gốc của người dùng không?

Trả về DUY NHẤT JSON:
{{
  "pronoun_resolution": 0.0,
  "standalone_quality": 0.0,
  "semantic_completeness": 0.0,
  "comment": "nhận xét ngắn"
}}

Payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


