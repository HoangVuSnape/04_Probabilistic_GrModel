# 05. NHẬT KÝ THỰC NGHIỆM & ĐỐI SOÁT KẾT QUẢ (EXPERIMENT LOGS & TRACKING)
> **Dự án:** Causal Bayesian Networks with Deep Learning for Educational Interventions  
> **Bộ dữ liệu:** Predict Students' Dropout and Academic Success (UCI ID: 697)  
> **Mục tiêu file:** Ghi vết (tracking) toàn bộ các đợt chạy thực nghiệm, cấu hình tham số, kết quả đo đạc và các phát hiện lỗi kỹ thuật.

---

## 1. Mẫu Nhật ký Thực nghiệm Chuẩn (Experiment Log Template)

Mỗi lần chạy thử nghiệm hoặc huấn luyện mô hình, ghi nhận thông tin theo mẫu:

```markdown
### [LOG-YYYYMMDD-STT] - Mô hình: [B0 / H1 / H2 / A1 / A2 / A3 / Synthetic]
* **Thời gian:** YYYY-MM-DD HH:MM
* **Hạt giống (Seed):** [ví dụ: 42]
* **Môi trường & Phần cứng:** Python 3.12 / PyTorch / pgmpy / DoWhy | CPU / GPU
* **Cấu hình siêu tham số:**
  * Estimator / Architecture: ...
  * Prior / Learning Rate / Batch Size: ...
  * Threshold / Mask: ...
* **Kết quả đo lường:**
  * P(Y=1 | do(T=0)): ...
  * P(Y=1 | do(T=1)): ...
  * ATE: ... (95% CI: [..., ...])
  * Dự báo Test: AUROC = ..., Macro-F1 = ..., Brier = ...
  * Thời gian chạy: ... giây
* **Kiểm định DoWhy Refutation:**
  * Placebo Refuter p-value: ...
  * Data Subset Refuter mean: ...
* **Ghi chú & Lỗi phát sinh (nếu có):** ...
```

---

## 2. Nhật ký Giai đoạn Khởi tạo (Phase 0: Project Setup)

### [LOG-20261029-001] — Khởi tạo Dự án & Chốt Đề tài Nghiên cứu
* **Thời gian:** 2026-10-29 13:30
* **Người thực hiện:** Nhóm nghiên cứu
* **Nội dung công việc:**
  1. Đọc và phân tích toàn bộ 62 slide hướng dẫn môn học `HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`.
  2. Tiếp nhận và tích hợp các dặn dò sống còn của Giảng viên:
     * Quy chế Midterm: Thời gian 2 tuần, nhóm 1–2 người, báo cáo trực tuyến, cơ chế duyệt Accept/Reject, quy tắc chênh lệch điểm $\text{First Author} - \text{Co-Author} = 1.0$.
     * Quy chế Final: Báo cáo bằng LaTeX PDF, kiểm tra tỷ lệ AI qua hệ thống Turnitin.
     * Nguyên tắc khoa học: Đo lường can thiệp $P(Y \mid \text{do}(T))$, không đo quan sát; tính toán kỳ vọng, rủi ro và độ bất định; chạy lặp ít nhất 5 seeds; học đồ thị bằng gradient liên tục (DAG-GNN/NOTEARS); tối thiểu 3 mô hình B0, H1, H2 (mở rộng H1+H2).
  3. **Chốt Bộ dữ liệu:** **Predict Students' Dropout and Academic Success (UCI ID: 697)** với $N = 4,424$ dòng, 36 thuộc tính.
  4. **Chốt Bài toán Nhân quả:**
     * Can thiệp ($T$): Cấp học bổng / hỗ trợ tài chính (`Scholarship holder`).
     * Kết quả ($Y$): Tốt nghiệp đại học thành công (`Target`: $1 = \text{Graduate}, 0 = \text{Dropout}$).
     * Biến nền ($Z$): Điểm tuyển sinh, nợ học phí, đóng học phí đúng hạn, học vấn phụ huynh, tuổi nhập học.
  5. Thiết lập hệ thống tài liệu nghiên cứu chuẩn tắc trong thư mục `research/` với tiền tố `01_` đến `05_`.
* **Trạng thái:** Hoàn tất giai đoạn khởi tạo. Sẵn sàng cho Giai đoạn 1 (Lập trình Data Pipeline và Synthetic Benchmark).

---

## 3. Bảng Theo dõi Kết quả 5 Seeds (Tracking Table)

*Bảng này sẽ được cập nhật liên tục khi chạy thực nghiệm:*

| Seed | Tập dữ liệu | Mô hình | $P(Y \mid \text{do}(T=0))$ | $P(Y \mid \text{do}(T=1))$ | Ước lượng ATE | Sai số vs Chân lý (nếu có) | AUROC (Test) | Runtime (s) | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | Synthetic | Chân lý | 0.300 | 0.600 | 0.300 | 0.000 | - | - | Reference |
| **42** | Synthetic | B0 | *Pending* | *Pending* | *Pending* | *Pending* | - | - | - |
| **42** | Synthetic | H1 | *Pending* | *Pending* | *Pending* | *Pending* | - | - | - |
| **42** | Synthetic | H2 | *Pending* | *Pending* | *Pending* | *Pending* | - | - | - |
| **42** | UCI ID 697| B0 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **42** | UCI ID 697| H1 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **42** | UCI ID 697| H2 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **123**| UCI ID 697| B0/H1/H2 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **456**| UCI ID 697| B0/H1/H2 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **789**| UCI ID 697| B0/H1/H2 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
| **999**| UCI ID 697| B0/H1/H2 | *Pending* | *Pending* | *Pending* | N/A | *Pending* | *Pending* | - |
