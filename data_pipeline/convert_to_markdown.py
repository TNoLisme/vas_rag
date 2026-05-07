import os
import re
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

class UniversalDocumentParser:
    def __init__(self):
        # Regex H5: Bắt số thứ tự (01., 13., A1., 1.1.)
        self.h5_regex = r'^([A-Z]?\d+([\.\)]|(\.\d+)+\.?))\s*(.*)'

    def get_true_level(self, para):
        """
        Lấy cấp độ thực sự của Paragraph từ XML (Outline Level).
        Level 0 trong XML = Heading 1, Level 1 = Heading 2...
        """
        try:
            pPr = para._element.pPr
            if pPr is not None:
                outline_lvl = pPr.xpath("./w:outlineLvl")
                if outline_lvl:
                    # Lấy giá trị val (0, 1, 2...)
                    val = int(outline_lvl[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"))
                    if 0 <= val <= 3: # Chỉ lấy từ H1 đến H4
                        return val + 1
        except:
            pass
        
        # Fallback: Nếu không có Outline Level, kiểm tra qua tên Style truyền thống
        style_name = para.style.name.lower()
        if any(h in style_name for h in ['heading 1', 'tiêu đề 1', 'h1']): return 1
        if any(h in style_name for h in ['heading 2', 'tiêu đề 2', 'h2']): return 2
        if any(h in style_name for h in ['heading 3', 'tiêu đề 3', 'h3']): return 3
        return None

    def is_h4_bold(self, para):
        """Nhận diện H4: Dòng in đậm, không phải số thứ tự, không phải Style H1-H3"""
        if not para.runs or not para.text.strip(): return False
        bold_text = "".join([run.text for run in para.runs if run.bold])
        full_text = para.text.strip()
        # Nếu >80% in đậm và không khớp Regex số (H5)
        return (len(bold_text) / len(full_text) > 0.8) and \
               not re.match(self.h5_regex, full_text) and \
               len(full_text) < 200

    def process_file(self, input_path, output_path):
        doc = Document(input_path)
        md_lines = []
        
        for element in doc.element.body:
            if element.tag.endswith('p'):
                para = Paragraph(element, doc)
                text = para.text.strip()
                if not text: continue

                true_level = self.get_true_level(para)
                if true_level:
                    md_lines.append(f"\n{'#' * true_level} {text}")
                    continue

                h5_match = re.match(self.h5_regex, text)
                if h5_match:
                    num = h5_match.group(1) 
                    body = h5_match.group(4) 
                    md_lines.append(f"\n##### {num}")
                    if body: md_lines.append(body)
                    continue

                if self.is_h4_bold(para):
                    md_lines.append(f"\n#### {text}")
                    continue

                md_lines.append(text)

            elif element.tag.endswith('tbl'):
                md_lines.append(self.table_to_markdown(Table(element, doc)))

        content = "\n".join([line for line in md_lines if line.strip()])
        content = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Đã cấu trúc hóa thành công (Sử dụng Outline Level): {os.path.basename(input_path)}")

    def table_to_markdown(self, table):
        md_table = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace('\n', '<br>') for cell in row.cells]
            md_table.append(f"| {' | '.join(cells)} |")
            if i == 0:
                md_table.append(f"| {' | '.join([':---:'] * len(cells))} |")
        return "\n" + "\n".join(md_table) + "\n"

if __name__ == "__main__":
    parser = UniversalDocumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    in_file = os.path.join(project_root, "data", "raw", "26.docx")
    out_file = os.path.join(project_root, "data", "processed", "26.md")
    parser.process_file(in_file, out_file)