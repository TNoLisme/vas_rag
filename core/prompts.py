from langchain_core.prompts import ChatPromptTemplate


ORDERED_METADATA_KEYS = ["Standard", "Chapter", "Section", "Article", "Point"]


def format_metadata_path(metadata, separator=" > "):
    return separator.join(
        str(metadata.get(key))
        for key in ORDERED_METADATA_KEYS
        if metadata.get(key)
    )


def build_context_text(docs):
    context_parts = []
    for index, doc in enumerate(docs, start=1):
        path = format_metadata_path(doc.metadata)
        context_parts.append(f"--- NGUỒN {index} ({path}) ---\n{doc.page_content}")
    return "\n\n".join(context_parts)


REWRITE_PROMPT_TEMPLATE = """
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
"""


REWRITE_PROMPT = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)


GENERATION_PROMPT_TEMPLATE = """Bạn là Chuyên gia Kế toán cao cấp. Hãy trả lời câu hỏi dựa trên các nguồn tri thức được cung cấp.
[TRI THỨC CUNG CẤP]
{context}
[CÂU HỎI]
{query}
[YÊU CẦU NGHIÊM NGẶT]
1. TRUNG THỰC: Chỉ sử dụng thông tin trong [TRI THỨC CUNG CẤP]. Không dùng kiến thức ngoài.
2. SUY LUẬN LOGIC: Nếu tài liệu không có câu trả lời trực tiếp, hãy dựa vào các nguyên tắc, định nghĩa trong nguồn tin để SUY LUẬN và giải quyết câu hỏi.
3. TRÍCH DẪN (CITATION): BẮT BUỘC ghi rõ nguồn ở cuối mỗi ý hoặc mỗi đoạn. Sử dụng tên chuẩn mực hoặc số Điều/Khoản có trong nhãn (Nguồn: ...).
   - Ví dụ: "Hàng tồn kho được tính theo giá gốc (Nguồn: CHUẨN MỰC SỐ 02 - HÀNG TỒN KHO > NỘI DUNG CHUẨN MỰC > 04.)".
4. PHONG CÁCH:
- Trình bày dạng văn xuôi mạch lạc hoặc danh sách gạch đầu dòng (bullet points).
- TUYỆT ĐỐI KHÔNG sử dụng tiêu đề Markdown (không dùng dấu #, ##, ###).
- KHÔNG bôi đậm dòng để làm tiêu đề giả. Câu trả lời phải phẳng và dễ đọc như một bản ghi chú chuyên môn.
CÂU TRẢ LỜI:"""


def build_generation_prompt(context_text, user_query):
    return GENERATION_PROMPT_TEMPLATE.format(context=context_text, query=user_query)


def build_sufficiency_check_prompt(context_text: str, user_query: str) -> str:
    return f"""[BỐI CẢNH] Bạn là trợ lý phân tích dữ liệu cho hệ thống RAG kế toán.
[NHIỆM VỤ] Đánh giá xem TÀI LIỆU có cung cấp ĐỦ CƠ SỞ để AI suy luận và trả lời CÂU HỎI hay không.
[TÀI LIỆU]
{context_text}
[CÂU HỎI]
{user_query}
[TIÊU CHÍ ĐÁNH GIÁ]
- Trả về YES: Nếu tài liệu nói trực tiếp về chủ đề của câu hỏi và chứa các dữ liệu, quy định hoặc khái niệm liên quan mà từ đó AI có thể tổng hợp thành câu trả lời hoàn chỉnh.
- Trả về NO: Chỉ khi tài liệu hoàn toàn KHÔNG liên quan, nội dung trọng tâm không nói về nội dung câu hỏi cần trả lời, nói về một chủ đề khác, hoặc không chứa bất kỳ từ khóa/nguyên tắc nào liên quan đến câu hỏi.
Lưu ý: Bạn không cần đáp án hoàn chỉnh, chỉ cần có đủ \"nguyên liệu\" liên quan đến nội dung của câu hỏi để suy luận.
CHỈ TRẢ VỀ: YES hoặc NO.
[KẾT QUẢ]:"""


def build_refine_search_query_prompt(user_query: str) -> str:
    return f"""Bạn là chuyên gia tra cứu văn bản pháp luật.
Lần tìm kiếm trước cho câu hỏi "{user_query}" không mang lại đủ thông tin.

[NHIỆM VỤ] Hãy viết lại một câu truy vấn tìm kiếm mới:
- Sử dụng các thuật ngữ kế toán đồng nghĩa (Ví dụ: "ghi nhận" thay cho "tính", "định khoản" thay cho "hạch toán").
- Nếu có mã chuẩn mực (VAS) hoặc thông tư, hãy giữ nguyên.
- Mục tiêu: Tìm được đúng đoạn văn bản chứa quy định chi tiết.

[TRẢ VỀ]: Duy nhất câu truy vấn mới, không giải thích."""


def build_nli_check_prompt(context_text: str, answer: str) -> str:
    return f"""[VAI TRÒ] Bạn là Kiểm toán viên nội bộ chuyên bắt lỗi ảo giác thông tin.
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


def build_correction_prompt(context_text: str, user_query: str, answer: str) -> str:
    return f"""[CẢNH BÁO] Câu trả lời trước đó đã vi phạm kiểm tra tính trung thực (NLI) và chứa thông tin ảo giác không có trong tài liệu.
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
4. Nếu tài liệu vẫn không đủ ý, hãy thừa nhận \"Tài liệu hiện tại không đề cập đến...\" thay vì suy diễn.
CÂU TRẢ LỜI ĐÃ VIẾT LẠI:"""

