# BẢNG CHECKLIST KIỂM THỬ VÀ NGHIỆM THU ADR-007
> **Tài liệu kiểm định cho:** `report/decisions/ADR_007_kaggle_s4e6_large_scale_analysis_and_causal_generalizability.md`  
> **Áp dụng cho:** File Notebook 6 `notebook/eda_and_causal_benchmark_kaggle_s4e6.ipynb`, Script thực nghiệm `tools/run_kaggle_s4e6_analysis.py`, và các artifacts kết quả  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Quy chuẩn:** Kaggle Playground Series s4e6, Slide 70–73, 80, 86 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).

---

## I. THÔNG TIN QUY CHUẨN & PHẠM VI NGHIỆM THU

- [x] **Mã quyết định kiến trúc:** ADR-007.
- [x] **Tên quyết định:** Kiểm toán Dữ liệu Quy mô Lớn Kaggle S4E6 (76,518 dòng), Học Cấu trúc Đồ thị Nhân quả và Đối chuẩn Tổng quát hóa Chuyển dịch.
- [x] **Tập dữ liệu áp dụng:** Kaggle Playground Series s4e6 ($N = 76,518$ dòng, sau lọc nhãn nhị phân $N = 61,578$ dòng) và UCI 697 ($N = 4,424 \to 3,630$ dòng).
- [x] **File notebook thực thi chính thức:** `notebook/eda_and_causal_benchmark_kaggle_s4e6.ipynb` (độc lập, không ghi đè, chạy thành công 100%).
- [x] **File thực thi kiểm chứng tự động:** `tools/run_kaggle_s4e6_analysis.py`.

---

## II. CHECKLIST CHI TIẾT 8 HẠNG MỤC KỸ THUẬT

### 1. Kiểm tra Dữ liệu & Phân lập Mẫu Nhị phân
- [x] **Kích thước dữ liệu gốc:** Tải thành công `data/kaggle_s4e6/train.csv` với $76,518$ dòng và 38 thuộc tính.
- [x] **Phân bố nhãn 3 lớp:** Graduate $36,282$ ($47.4\%$), Dropout $25,296$ ($33.1\%$), Enrolled $14,940$ ($19.5\%$).
- [x] **Lọc bỏ cohort Enrolled:** Thu được $N_{\text{clean}} = 61,578$ dòng với $Y \in \{0, 1\}$ (Tỷ lệ tốt nghiệp: $58.92\%$).

### 2. Kiểm toán Độ Trung thực Phân phối (KS-test & Wasserstein)
- [x] **Kiểm định 2-mẫu KS-test:** Thực hiện trên toàn bộ các biến then chốt ($Z_1 \dots Z_5, T$).
- [x] **Khoảng cách Wasserstein:** Xác nhận độ lệch nhỏ: Tuổi ($0.95$), Điểm tuyển sinh ($1.71$), Nợ học phí ($0.0357$).
- [x] **Tỷ lệ can thiệp $T$ (Học bổng):** UCI đạt $26.69\%$, Kaggle đạt $27.51\%$ (Độ lệch tuyệt đối $< 0.82\%$).
- [x] **Tỷ lệ học phí $Z_5$:** UCI đạt $86.61\%$, Kaggle đạt $87.37\%$ (Độ lệch tuyệt đối $< 0.76\%$).
- [x] **Lưu trữ bảng đối chuẩn:** [`Table_Kaggle_S4E6_Comparison.csv`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Kaggle_S4E6_Comparison.csv).

### 3. Kiểm tra Học Cấu trúc Đồ thị Nhân quả Quy mô Lớn (NOTEARS on 61,578 Rows)
- [x] **Ràng buộc thời gian:** Áp dụng Ma trận Cấm Cạnh thời gian (Forbidden Mask $M$).
- [x] **Cấu trúc đồ thị học được:** 7 nút, 8 cạnh có hướng, xác thực `is_directed_acyclic_graph == True`.
- [x] **Cung nhân quả cốt lõi:** Xác nhận học thành công cung trực tiếp $T \to Y$.
- [x] **Tập cha trực tiếp của Y:** $\text{Pa}(Y) = \{Z_1, Z_5, T\}$ (Tuổi, Đóng học phí và Học bổng).

### 4. Kiểm tra Ước lượng Can thiệp (ATE) trên Kaggle S4E6
- [x] **Phân chia mẫu lớn:** 70% Train ($43,104$ SV) / 30% Test ($18,474$ SV).
- [x] **Mô hình B0 trên Kaggle:** $ATE = \mathbf{+0.3040}$ ($Risk(do(0)) = 51.20\%$, $Risk(do(1)) = 81.60\%$).
- [x] **Mô hình H2 trên Kaggle:** $ATE = \mathbf{+0.3031}$ ($Risk(do(0)) = 51.09\%$, $Risk(do(1)) = 81.39\%$).
- [x] **Khoảng tin cậy 95% Bootstrap (1,000 resamples):** $[+0.3016, +0.3061]$ (Biên sai số cực nhỏ $\pm 0.0022$).

### 5. Kiểm tra 4 Kịch bản Chuyển dịch Chéo (Cross-Dataset Transfer Benchmark)
- [x] **Kịch bản 1 (In-domain UCI):** Train UCI $\to$ Test UCI: AUROC $0.8039$, Brier $0.1761$, $ATE = +0.2153$.
- [x] **Kịch bản 2 (Thực tế $\to$ Synthetic):** Train UCI ($3,630$ SV) $\to$ Test Kaggle ($61,578$ SV): AUROC $\mathbf{0.8537}$, Brier $\mathbf{0.1539}$, $ATE = \mathbf{+0.2143}$.
- [x] **Kịch bản 3 (Synthetic $\to$ Thực tế):** Train Kaggle ($61,578$ SV) $\to$ Test UCI ($3,630$ SV): AUROC $0.8120$, Brier $0.1707$, $ATE = +0.3217$.
- [x] **Kịch bản 4 (In-domain Kaggle):** Train Kaggle $\to$ Test Kaggle: AUROC $0.8546$, Brier $0.1472$, $ATE = +0.3040$.
- [x] **Lưu trữ bảng chuyển dịch:** [`Table_Cross_Dataset_Transfer.csv`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Cross_Dataset_Transfer.csv).

### 6. Kiểm tra Hiệu ứng Can thiệp Không đồng nhất (CATE) trên 61k Dòng
- [x] **Phân tầng Độ tuổi:**
  - Nhóm lớn tuổi ($Z_1 \ge 1$, $N = 28,797$): $CATE = \mathbf{+0.3745}$ (Relative Risk $2.11\times$).
  - Nhóm trẻ tuổi ($Z_1 = 0$, $N = 32,781$): $CATE = \mathbf{+0.2402}$ (Relative Risk $1.36\times$).
- [x] **Phân tầng Nợ học phí:**
  - Nhóm có nợ ($Z_4 = 1$, $N = 4,809$): Tỷ lệ tốt nghiệp tăng từ $10.05\%$ lên $25.41\%$ (Relative Risk **$2.53\times$**).
- [x] **Lưu trữ bảng CATE Kaggle:** [`Table_Kaggle_Causal_Estimates.csv`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Kaggle_Causal_Estimates.csv).

### 7. Kiểm tra Toàn vẹn File Notebook 6 (`eda_and_causal_benchmark_kaggle_s4e6.ipynb`)
- [x] **Tính độc lập:** Không ghi đè lên các Notebook 1, 2, 3, 4, 5.
- [x] **Cấu trúc cells:** Gồm 13 cells (6 markdown khoa học, 7 code cells chuẩn mực).
- [x] **Kiểm thử tự động:** Đã chạy qua `tools/test_notebook_execution.py`, toàn bộ 7 code cells đều trả về `SUCCESS` (Exit code: 0).

### 8. Kiểm tra Lưu trữ Biểu đồ Học thuật Đối chứng
- [x] **Hình 6 (Distributional Audit & Fidelity):** `artifacts/figures/fig6_uci_vs_kaggle_distributions.png` & `.svg`.
- [x] **Hình 7 (Cross-Dataset Generalization):** `artifacts/figures/fig7_cross_dataset_causal_generalization.png` & `.svg`.

---

## III. KẾT LUẬN NGHIỆM THU

Toàn bộ các tiêu chí kỹ thuật, kiểm định phân phối, và độ tin cậy suy luận can thiệp của **ADR-007** trên tập dữ liệu lớn Kaggle S4E6 ($76,518$ dòng) đã được kiểm tra và nghiệm thu **ĐẠT 100%**. Kết quả đã xác thực khả năng tổng quát hóa xuất sắc của các mô hình đồ thị xác suất trong đề tài.
