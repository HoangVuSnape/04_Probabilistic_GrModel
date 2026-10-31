# TRUNG TÂM BÁO CÁO, QUYẾT ĐỊNH & FILE NHẬT KÝ THỰC THI (REPORT & LOGS HUB)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện  
> **Chức năng:** Quản lý tập trung các quyết định nghiên cứu (Decisions / ADRs), các file nhật ký thực thi dạng thô (`.log`), cùng toàn bộ tài liệu báo cáo Midterm & Final.

---

## 1. Mục đích của Thư mục `report/`

Thư mục `report/` đóng vai trò là **kho lưu trữ văn bản chính thức và bằng chứng kiểm toán (Audit Trail)** của dự án, phục vụ trực tiếp cho việc nộp bài và bảo vệ trước Giảng viên / Hội đồng:
1. **Lưu trữ các quyết định then chốt (Decisions):** Ghi nhận lý do khoa học tại sao chọn bộ dữ liệu, tại sao chọn cấu trúc đồ thị, tại sao áp đặt các ràng buộc cấm cạnh (Domain mask) để sẵn sàng trả lời chất vấn (Q&A).
2. **Lưu trữ file log máy đọc (`.log`):** Toàn bộ stdout/stderr khi chạy huấn luyện PyTorch, quá trình tối ưu gradient của DAG-GNN, kết quả chạy 100 lần mô phỏng DoWhy Refutation được xuất ra file log thô để đối soát.
3. **Quản trị báo cáo Midterm & Final:** Nơi soạn thảo slide, tài liệu thuyết minh giữa kỳ và toàn bộ mã nguồn bài báo khoa học LaTeX cuối kỳ.

---

## 2. Cấu trúc Phân mục bên trong `report/`

```text
report/
├── README.md                      <-- File này: Tổng quan chức năng và quy chuẩn lưu trữ
├── decisions/                     <-- Các bản ghi quyết định kiến trúc & nghiên cứu (ADRs)
│   ├── ADR_001_dataset_selection.md  <-- Quyết định chọn bộ dữ liệu UCI ID 697
│   └── ADR_002_causal_formulation.md <-- Quyết định thiết kế biến can thiệp T, kết quả Y, biến nền Z
├── logs/                          <-- Nơi chứa các file log thực thi thực nghiệm (.log)
│   ├── .gitkeep
│   └── experiment_template.log       <-- Mẫu định dạng ghi log thực nghiệm
├── midterm/                       <-- Báo cáo, Slide và Tài liệu thuyết minh Giữa kỳ
│   └── README.md                  <-- Quản lý tài liệu thuyết trình & kịch bản demo Midterm
└── final/                         <-- Báo cáo Khoa học Toàn văn Cuối kỳ (LaTeX & PDF)
    └── README.md                  <-- Quản lý bài báo khoa học LaTeX, hình ảnh và kiểm duyệt Turnitin
```

---

## 3. Quy chuẩn Quản lý File Log (`report/logs/`)

Mọi script Python thực thi (huấn luyện, đánh giá, kiểm thử refutation) đều được khuyến nghị cấu hình ghi log song song ra console và file:
* **Tên file log chuẩn:** `YYYYMMDD_HHMMSS_<model_or_task>_<seed>.log`
  * Ví dụ: `20261102_143000_b0_baseline_seed42.log`
  * Ví dụ: `20261105_091520_h1_mlp_seed42.log`
  * Ví dụ: `20261108_162000_dowhy_refutation_seed42.log`
* **Nội dung trong log:** Ghi rõ cấu hình siêu tham số, thời gian bắt đầu/kết thúc (runtime timestamp), tiến độ loss qua từng epoch, ma trận trọng số kề học được, và kết quả kiểm định p-value.

---

## 4. Điều hướng nhanh

* 📑 [Báo cáo Tổng hợp Dự án Toàn diện (FINAL_COURSE_PROJECT_REPORT.md)](./FINAL_COURSE_PROJECT_REPORT.md)
* 📊 [Báo cáo Phân tích Chuyên sâu Hình ảnh Thực nghiệm (NOTEBOOK_VISUAL_ANALYSIS_REPORT.md)](./NOTEBOOK_VISUAL_ANALYSIS_REPORT.md)
* 📋 [Bản ghi Quyết định Kiến trúc & Bảng Kiểm tra (Decisions / ADRs & Checklists)](./decisions/)
* 📜 [Thư mục File Log Thực nghiệm](./logs/)
* 🎤 [Báo cáo & Slide Midterm](./midterm/README.md)
* 📑 [Bài báo Khoa học LaTeX Cuối kỳ (Final)](./final/README.md)
