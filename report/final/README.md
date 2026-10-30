# KHU VỰC BÁO CÁO KHOA HỌC CUỐI KỲ (FINAL REPORT - LATEX & TURNITIN)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Định dạng bắt buộc:** **File PDF biên dịch từ LaTeX** (kèm toàn bộ mã nguồn `.tex`)  
> **Kiểm duyệt học thuật:** **Turnitin Check AI / Plagiarism**  
> **Thang điểm:** 10 điểm (chia đều cho 9 mục chuẩn mực 2.1 – 2.9)  

---

## 1. Cấu trúc Thư mục Bài báo Khoa học LaTeX

Thư mục này được tổ chức chuyên nghiệp như một dự án bài báo khoa học chuẩn quốc tế (IEEE / Springer / ACM):

```text
report/final/
├── README.md                      <-- File này: Quy chuẩn bài báo và kiểm duyệt Turnitin
├── main.tex                       <-- Mã nguồn chính LaTeX (bám sát report_template_structure.md)
├── references.bib                 <-- File trích dẫn tài liệu tham khảo BibTeX (Pearl, DoWhy, pgmpy, DAG-GNN, UCI)
├── figures/                       <-- Chứa ảnh sơ đồ DAG, biểu đồ huấn luyện loss, biểu đồ ATE
│   ├── dag_b0_domain.png
│   ├── dag_h2_learned.png
│   └── refutation_distributions.png
└── tables/                        <-- Các bảng số liệu LaTeX (Bảng A, Bảng B, Bảng C)
    ├── table_a_uci_results.tex
    ├── table_b_synthetic_results.tex
    └── table_c_ablations.tex
```

---

## 2. Tiêu chuẩn Kiểm duyệt Turnitin Check AI & Đạo văn

Để bài báo đạt điểm tối đa và không bị gắn cờ vi phạm liêm chính học thuật:
1. **Tự viết diễn giải khoa học (Authentic Academic Writing):**
   * Các phần lập luận, phân tích bảng số liệu thực nghiệm Bảng A, B, C và thảo luận hạn chế phải do nhóm tự viết dựa trên kết quả đo lường thực tế từ mã nguồn.
   * Tuyệt đối không sao chép nguyên văn văn bản do ChatGPT hoặc LLMs sinh ra vào bài báo.
2. **Khai báo Sử dụng AI (AI-Use Statement):**
   * Đặt tại cuối bài báo (trước References).
   * Khai báo rõ ràng: Công cụ AI (Antigravity/ChatGPT/Claude) chỉ được sử dụng để hỗ trợ tra cứu cú pháp LaTeX, soát lỗi chính tả tiếng Anh hoặc định dạng bảng biểu; toàn bộ dữ liệu, thiết kế thực nghiệm và phân tích là công trình độc lập của nhóm.
3. **Trích dẫn BibTeX đầy đủ:**
   * Mọi định nghĩa về Pearl's do-calculus, thuật toán NOTEARS/DAG-GNN, thư viện `DoWhy`, `pgmpy` và bộ dữ liệu UCI ID 697 đều phải có trích dẫn chuẩn (`\cite{...}`).

---

## 3. Lệnh Biên dịch LaTeX sang PDF

Nhóm có thể biên dịch bài báo trên máy cục bộ bằng lệnh:
```powershell
# Biên dịch tài liệu đầy đủ kèm trích dẫn BibTeX
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Hoặc đưa toàn bộ thư mục `report/final/` lên nền tảng **Overleaf** để biên dịch trực tuyến.
