# -*- coding: utf-8 -*-
"""Ghép KICH_BAN_CHON_LOC.md bám notebook hiện tại (đã bỏ mục phân bổ chính sách và Kaggle).
Cell mục 1–7 và kết luận lấy từ kịch bản đầy đủ; cell mục 8–9 lấy từ bản giải thích sâu; đầu mục 9 có phần tổng quan."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'report' / 'midterm'
NB = REPO / 'notebook' / '00_MASTER_Causal_PGM_Educational_Intervention.ipynb'
TONG = len(json.load(open(NB, encoding='utf-8'))['cells'])
full = (ROOT / 'KICH_BAN_TRINH_BAY_NOTEBOOK.md').read_text(encoding='utf-8')
deep = (ROOT / 'kich_ban_sau_muc_8_9.md').read_text(encoding='utf-8')


def split_cells(text):
    out = {}
    for p in re.split(r'(?m)^(?=### Cell \d+ · )', text):
        m = re.match(r'### Cell (\d+) · ', p)
        if m:
            out[int(m.group(1))] = re.split(r'(?m)^(?:---\s*$|## )', p)[0].rstrip()
    return out


def rep(text, a, b):
    assert a in text, f'không tìm thấy: {a[:70]}'
    return text.replace(a, b)


def khop_notebook_moi(src_no, text):
    """Kịch bản đầy đủ viết theo bản 89 cell; sửa các chỗ nhắc mục 10, mục 11 và RQ6 cho khớp notebook hiện tại."""
    if src_no == 0:
        text = rep(text, 'Mục 9 và 10 là phần mở rộng. Mục 11 trả lời lại từng câu hỏi.',
                   'Mục 9 là phần mở rộng. Mục 10 trả lời lại từng câu hỏi.')
    if src_no == 3:
        text = rep(text, 'Ngoài ra có sáu câu phụ:', 'Ngoài ra có năm câu phụ:')
        text = rep(text, '* RQ5 và RQ6 là phần mở rộng: cơ chế tác động, và cách phân bổ học bổng.',
                   '* RQ5 là phần mở rộng: học bổng tác động qua kênh nào và nhóm nào hưởng lợi.')
        text = rep(text, 'Ở mục 11, em quay lại', 'Ở mục 10, em quay lại')
    if src_no == 88:
        text = rep(text, '### Cell 88 · markdown · 11.1 Trả lời RQ + 11.2 Khuyến nghị',
                   '### Cell 85 · markdown · 10.1 Trả lời RQ + 10.2 Khuyến nghị')
        text = rep(text, '* **RQ6:** ủng hộ có điều kiện, cần thí điểm.\n', '')
    return text


old, new = split_cells(full), split_cells(deep)
RENUM = {88: 85}  # notebook hiện tại: Tổng kết là cell 85 (đã bỏ 3 cell mục phân bổ chính sách)
KEEP_OLD = [0, 2, 3, 4, 8, 9, 11, 12, 13, 14, 17, 18, 22, 23, 25, 30, 31, 35, 36, 48, 88]
DEEP = [37, 41, 43, 45, 47, 49, 52, 53, 57, 58, 59, 61, 62, 64, 69, 71, 73, 75, 77, 79, 83]
order = sorted([(RENUM.get(c, c), c, 'old') for c in KEEP_OLD] + [(c, c, 'deep') for c in DEEP])
assert not ([c for c in KEEP_OLD if c not in old] + [c for c in DEEP if c not in new])

SEC = {0: 'MỞ ĐẦU', 2: 'MỤC 1 — LÝ DO, CÂU HỎI, DỮ LIỆU', 17: 'MỤC 2 — CHÊNH LỆCH QUAN SÁT THÔ',
       22: 'MỤC 3 — HỌC CẤU TRÚC (MINH HỌA)', 25: 'MỤC 4 — HỌC THAM SỐ: MLE VÀ BDEU',
       30: 'MỤC 5–6 — KIỂM TRA DAG VÀ DOMAIN DAG', 35: 'MỤC 7 — DOWHY',
       37: 'MỤC 8 — BA MÔ HÌNH B0, H1, H2 (PHẦN CHÍNH)', 69: 'MỤC 9 — PHÂN TÍCH MỞ RỘNG',
       85: 'MỤC 10 — KẾT LUẬN'}
head = f'''# Kịch bản nói — các cell cần trình bày

Bản chọn lọc **{len(order)} / {TONG} cell**, đi theo đúng thứ tự notebook (số cell giống notebook).
* **Mục 1–7 và 10 (kết luận):** 🎤 Nói, 📘 Từ khóa mới, ⚠️ Lưu ý.
* **Mục 8–9:** 🧩 Cell này làm gì → 🎤 Nói → 📊 Đọc output (đọc output khi chỉ vào màn hình).
* **Mục 9** có thêm phần **Tổng quan** ở đầu mục: mục tiêu, input, output và ý nghĩa của từng phần 9.A đến 9.H.

**Các cell được lướt qua, không cần nói riêng:**

| Cell | Lý do bỏ |
| :--- | :--- |
| 1, 5, 6, 7 | Tuyên bố dùng AI (đã có Vòng 0 trong QA), hướng dẫn cài đặt, in môi trường, đọc lại output cell 4 |
| 10, 15, 16, 19 | Markdown giới thiệu hoặc tóm tắt — nội dung đã nói khi trình bày cell code bên cạnh |
| 20, 21, 24, 26, 27, 28, 29, 32, 33, 34 | Phần lý thuyết dạng chữ và cell "đọc kết quả" nhắc lại output — gộp vào lời nói của cell hình/bảng cùng mục |
| 38, 39, 40, 42 | Mở đầu pipeline, nạp thư viện, mô tả dữ liệu 7 biến (đã có ở bảng từ điển biến) |
| 44, 46, 50, 51 | Markdown thiết kế B0 / H1 / giao thức đánh giá và hàm đánh giá dùng chung — đã nói trong cell code 45, 47, 52 |
| 54, 55, 56, 60, 63, 65 | Markdown đọc kết quả và cell xuất CSV — đã gộp vào cell 53, 58, 59, 61 |
| 66, 67, 68, 70, 72, 74, 76, 78, 80, 81, 82, 84 | Markdown mở đầu / đọc kết quả của mục 9 và bảng tổng hợp — đã gộp vào cell code tương ứng |

Notebook hiện tại đã bỏ phần phân bổ học bổng và chuyển sang Kaggle; mục **Tổng kết** là mục 10 (cell 85).
Câu hỏi chất vấn: tab **QA chất vấn**. Từ khóa khó: tab **Glossary** hoặc **Công thức**. Bản đủ từng cell: tab **Kịch bản đầy đủ** (đánh số theo bản notebook cũ 89 cell).
'''


def tong_quan_muc_9() -> str:
    """Đọc muc_9_tong_quan.md và hạ cấp tiêu đề: # -> ### (một thẻ), ## -> #### (nằm trong thẻ đó)."""
    t = (ROOT / 'muc_9_tong_quan.md').read_text(encoding='utf-8')
    t = re.sub(r'(?m)^## ', '#### ', t)
    t = re.sub(r'(?m)^# .*$', '### Tổng quan mục 9 — mục tiêu, input, output, ý nghĩa từng phần', t, count=1)
    return t.rstrip() + '\n'


out = [head]
for disp, src, kind in order:
    if disp in SEC:
        out.append(f'\n---\n\n## {SEC[disp]}\n')
        if disp == 69:
            out.append(tong_quan_muc_9())  # tổng quan mục 9 đặt ngay đầu mục, trước Cell 69
    body = new[src] if kind == 'deep' else khop_notebook_moi(src, old[src])
    out.append(body + '\n')
(ROOT / 'KICH_BAN_CHON_LOC.md').write_text('\n'.join(out), encoding='utf-8')
print('KICH_BAN_CHON_LOC.md:', len(order), 'cell /', TONG)
