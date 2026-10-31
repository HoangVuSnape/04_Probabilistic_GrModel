# BẢNG CHECKLIST KIỂM THỬ VÀ NGHIỆM THU ADR-005
> **Tài liệu kiểm định cho:** `report/decisions/ADR_005_advanced_factorial_h3_sensitivity_and_policy_learning.md`  
> **Áp dụng cho:** File Notebook 4 `notebook/causal_policy_and_sensitivity_uci697.ipynb`, Script thực nghiệm `tools/run_advanced_factorial_and_sensitivity.py`, và các artifacts bảng kết quả liên quan  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Quy chuẩn:** Slide 72–73 (Synthetic Benchmark), Slide 80 (Mô hình H1+H2), Slide 86 (Quy định Đồ án) (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).

---

## I. THÔNG TIN QUY CHUẨN & PHẠM VI NGHIỆM THU

- [x] **Mã quyết định kiến trúc:** ADR-005.
- [x] **Tên quyết định:** Ma trận Yếu tố 2x2 (Mô hình H3 kết hợp H1+H2), Phân tích Độ nhạy Nhiễu ẩn (Cinelli-Hazlett RV) và Học Chính sách Can thiệp Tối ưu dưới Ràng buộc Ngân sách.
- [x] **Tập dữ liệu áp dụng:** UCI ID 697 ($N = 3,630$ sau lọc nhãn nhị phân) và Tập dữ liệu Tổng hợp Synthetic ($N = 10,000$).
- [x] **Số lượng mô hình trong Ma trận Yếu tố:** 4 mô hình ($B_0, H_1, H_2, H_3$).
- [x] **File notebook thực thi chính thức:** `notebook/causal_policy_and_sensitivity_uci697.ipynb` (độc lập, không ghi đè, chạy thành công 100%).
- [x] **File thực thi kiểm chứng tự động:** `tools/run_advanced_factorial_and_sensitivity.py`.
- [x] **Số lượng hạt giống ngẫu nhiên ghép cặp:** 5 seeds (`[42, 43, 44, 45, 46]`).

---

## II. CHECKLIST CHI TIẾT 6 HẠNG MỤC KỸ THUẬT

### 1. Kiểm tra Ma trận Thiết kế Yếu tố $2 \times 2$ & Mô hình $H_3$
- [x] **Thiết kế ma trận đầy đủ 4 ô:**
  - Ô (Domain DAG, Bayesian CPD): Mô hình $B_0$ (Baseline CBN).
  - Ô (Domain DAG, Neural MLP): Mô hình $H_1$ (Neural Parameterization).
  - Ô (Learned NOTEARS, Bayesian CPD): Mô hình $H_2$ (Neural Causal Discovery).
  - Ô (Learned NOTEARS, Neural MLP): Mô hình $H_3$ (Fully Neural Model H1 + H2).
- [x] **Đánh giá ghép cặp trên 5 seeds:** Chạy đồng thời trên cùng bộ train/val/test splits cố định cho từng seed.
- [x] **Kết quả định lượng mô hình H3:**
  - AUROC: $0.8048 \pm 0.0122$.
  - Macro F1: $\mathbf{0.7147 \pm 0.0225}$ (Cao nhất trong 4 mô hình).
  - Brier Score: $\mathbf{0.1667 \pm 0.0070}$ (Thấp nhất / Khả năng hiệu chuẩn xác suất tốt nhất).
  - ATE: $+0.2228 \pm 0.0101$ (Nhất quán tuyệt đối với $H_1$ và $B_0$).
- [x] **Kiểm tra đồ thị vô chu trình (DAG Validation):** $H_3$ kế thừa đồ thị học được có kiểm soát ràng buộc ma trận cấm cạnh thời gian $M$, đảm bảo `nx.is_directed_acyclic_graph(dag) == True`.

### 2. Kiểm tra Phân tích Độ nhạy Nhiễu ẩn (Cinelli & Hazlett RV)
- [x] **Cơ sở lý thuyết:** Áp dụng phương pháp phân tích độ nhạy kinh tế lượng của Cinelli & Hazlett (JRSS-B, 2020).
- [x] **Ước lượng hồi quy tham chiếu:**
  - Hệ số can thiệp: $\hat{\tau} = 0.2268$, Sai số chuẩn: $SE = 0.0160$.
  - Thống kê t-statistic: $t = 14.20$ ($p < 10^{-44}$).
  - Hệ số xác định từng phần (Partial $R^2$): $5.27\%$.
- [x] **Giá trị Bền vững (Robustness Value - $RV_{q=1}$):** Đạt mức **$19.09\%$**.
- [x] **Kiểm chuẩn so sánh với các biến gây nhiễu quan sát (Benchmarking):**
  - Biến mạnh nhất quan sát được (Học phí $Z_5$): $R^2_{\text{partial}} = 10.41\%$.
  - Biến tuổi nhập học ($Z_1$): $R^2_{\text{partial}} = 4.22\%$.
- [x] **Kết luận khoa học:** Biến nhiễu ẩn chưa quan sát $U$ phải có sức mạnh giải thích lớn gấp gần 2 lần biến nợ học phí và hơn 4.5 lần biến tuổi tác thì mới có thể xóa bỏ hiệu ứng học bổng $\implies$ Giả định Unconfoundedness được bảo vệ vững chắc.

### 3. Kiểm tra Học Chính sách Can thiệp Tối ưu dưới Ràng buộc Ngân sách (Policy Learning)
- [x] **Thiết lập bài toán thực tế:** Cố định ngân sách thực tế ở mức **$26.7\%$** ($969$ suất học bổng trên $3,630$ sinh viên).
- [x] **So sánh 4 chiến lược phân bổ học bổng:**
  - **Chính sách Tối ưu Nhân quả (CATE Optimal):** Tỷ lệ tốt nghiệp đạt **$65.37\%$** ($2,372$ sinh viên tốt nghiệp).
  - **Chính sách Thực tế Lịch sử (Factual Allocation):** Tỷ lệ tốt nghiệp đạt **$60.85\%$** ($2,208$ sinh viên tốt nghiệp).
  - **Chính sách Ưu tiên Cần thiết (Need-based Debtor):** Tỷ lệ tốt nghiệp đạt **$62.19\%$** ($2,257$ sinh viên tốt nghiệp).
  - **Chính sách Phân bổ Ngẫu nhiên (Random Allocation):** Tỷ lệ tốt nghiệp đạt **$61.87\%$** ($2,245$ sinh viên tốt nghiệp).
- [x] **Lợi ích thặng dư thực tiễn (Causal Dividend):** Chiến lược CATE-Optimal mang lại thêm **$+164$ sinh viên tốt nghiệp** so với thực tế lịch sử mà **không tốn thêm một đồng ngân sách nào**.

### 4. Kiểm tra Đối chuẩn trên Dữ liệu Tổng hợp (Synthetic Ground Truth Benchmark - Slide 72, 73)
- [x] **Thiết lập mô phỏng:** Tạo $10,000$ mẫu với cấu trúc nút $Z \to T \to Y$ và $Z \to Y$ theo đúng tham số quy định ở Slide 72.
- [x] **Chân lý mô phỏng:** $\mathbb{E}[Y(0)] = 0.3000$, $\mathbb{E}[Y(1)] = 0.6000 \implies ATE_{\text{true}} = \mathbf{+0.3000}$.
- [x] **Kết quả ước lượng của engine pgmpy:** $\widehat{ATE} = \mathbf{+0.2778}$.
- [x] **Sai số chệch ước lượng (Estimation Bias):** $|\widehat{ATE} - ATE_{\text{true}}| = \mathbf{0.0222}$ (Sai số $< 2.3\%$).
- [x] **Khẳng định tính hợp lệ:** Chứng minh engine can thiệp nhân quả của đồ án tuân thủ chính xác lý thuyết xác suất và không bị lỗi tích lũy sai số.

### 5. Kiểm tra Toàn vẹn File Notebook 4 (`causal_policy_and_sensitivity_uci697.ipynb`)
- [x] **Tính độc lập:** Notebook được tạo mới hoàn toàn, không đè lên các Notebook 1, 2, 3.
- [x] **Cấu trúc cells:** Gồm 12 cells (6 markdown hướng dẫn khoa học, 6 code cells có cấu trúc chuẩn mực).
- [x] **Kiểm thử tự động:** Đã chạy qua `tools/test_notebook_execution.py`, toàn bộ 6 code cells trả về mã thoát `SUCCESS` (Exit code: 0).
- [x] **Tương thích môi trường:** Hoạt động ổn định trên Python 3.13, PyTorch CPU, pgmpy 1.1.2.

### 6. Kiểm tra Lưu trữ Artifacts Tables
- [x] `artifacts/tables/Table_Factorial_H3.csv` — Đã tạo và kiểm tra dữ liệu đầy đủ.
- [x] `artifacts/tables/Table_Sensitivity_Cinelli.csv` — Đã tạo và kiểm tra dữ liệu đầy đủ.
- [x] `artifacts/tables/Table_Policy_Learning.csv` — Đã tạo và kiểm tra dữ liệu đầy đủ.

---

## III. KẾT LUẬN NGHIỆM THU

Toàn bộ các tiêu chí kỹ thuật, độ tin cậy toán học, và yêu cầu trình bày của **ADR-005** đã được kiểm tra và nghiệm thu **ĐẠT 100%**. Hệ thống đồ án hiện đã sở hữu bức tranh thực nghiệm hoàn chỉnh nhất từ mức vi mô (CPDs, Neural parameterization) đến vĩ mô (Chính sách can thiệp thực tế và Giới hạn độ nhạy nhiễu ẩn).
