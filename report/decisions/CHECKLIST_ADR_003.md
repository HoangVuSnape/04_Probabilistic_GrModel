# BẢNG CHECKLIST KIỂM THỬ VÀ NGHIỆM THU ADR-003
> **Tài liệu kiểm định cho:** `report/decisions/ADR_003_causal_modeling_and_ablation_architecture.md`  
> **Áp dụng cho:** File Notebook 2 `notebook/causal_modeling_uci697.ipynb` và Script thực nghiệm `tools/run_ablations_and_extensions.py`  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Quy chuẩn:** Slide 51, 53, 57–63, 64–67, 74–75, 80 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

---

## I. THÔNG TIN QUY CHUẨN & PHẠM VI NGHIỆM THU

- [x] **Mã quyết định kiến trúc:** ADR-003.
- [x] **Tên quyết định:** Kiến trúc mô hình hóa nhân quả (B0, H1, H2), Động cơ truy vấn Vector hóa $O(1)$ và Nghiên cứu bóc tách (Ablations).
- [x] **Tập dữ liệu áp dụng:** UCI ID 697 (Predict Students' Dropout and Academic Success).
- [x] **Giao thức thực nghiệm:** Đánh giá ghép cặp trên 5 hạt giống cố định: `seeds = [42, 43, 44, 45, 46]`.
- [x] **File notebook thực thi chính:** `notebook/causal_modeling_uci697.ipynb`.

---

## II. CHECKLIST CHI TIẾT 9 HẠNG MỤC KỸ THUẬT

### 1. Kiểm tra Tiền xử lý Dữ liệu & Đóng băng Phân chia (Section 7)
- [x] **Lọc đối tượng Enrolled:** Chỉ giữ nhóm `Dropout` và `Graduate`, binarize thành $Y \in \{0, 1\}$. Kích thước sau lọc: đúng **3,630 dòng** (loại bỏ 794 dòng Enrolled).
- [x] **Không rò rỉ biến tương lai (No Leakage):** Loại bỏ toàn bộ các biến tín chỉ học kỳ 1 và 2 (`Curricular units 1st/2nd sem`) khỏi tập điều chỉnh $\mathbf{Z}$.
- [x] **Tỷ lệ phân chia:** Chuẩn hóa 70% Train ($2,541$ dòng) / 15% Validation ($544$ dòng) / 15% Test ($545$ dòng).
- [x] **Đóng băng chỉ số (Frozen Splits):** Đã xuất và lưu trữ 5 file chỉ số dòng cố định tại `artifacts/splits/split_seed_{seed}.json`.
- [x] **Fitting cô lập:** Tiền xử lý chỉ fit trên tập Train của từng seed, biến Test hoàn toàn không bị rò rỉ phân phối.

### 2. Kiểm tra Mô hình B0 - Baseline CBN (Section 8)
- [x] **Cấu trúc Primary Domain DAG:** Đồ thị gồm đúng **7 nút** ($\{Z_1..Z_5, T, Y\}$) và **11 cạnh có hướng**.
- [x] **Bảo vệ trật tự thời gian:** Đảm bảo $Z \to T$, $Z \to Y$, $T \to Y$. Kiểm tra `nx.is_directed_acyclic_graph` trả về `True`.
- [x] **Bayesian Smoothing:** Ước lượng CPDs bằng prior BDeu với `equivalent_sample_size = 5`.
- [x] **Tương thích đa phiên bản pgmpy:** Sử dụng hàm bọc `fit_discrete_bayesian_model` hỗ trợ cả `DiscreteBayesianEstimator` (pgmpy >= 1.1) và `BayesianEstimator` (pgmpy legacy).
- [x] **Kiểm tra tính hợp lệ:** Gọi `model.check_model()` trả về `True` không có lỗi tổng xác suất $\neq 1$.

### 3. Kiểm tra Mô hình H1 - Hybrid Neural CPD (Section 9)
- [x] **Bảo tồn cấu trúc Markov:** Mạng MLP chỉ nhận đầu vào duy nhất là các nút cha trực tiếp của $Y$: $\text{Parents}(Y) = \{T, Z_1, Z_2, Z_3, Z_4, Z_5\}$ (8 chiều đặc trưng số học sau One-hot).
- [x] **Kiến trúc mạng sâu PyTorch:** Linear(8 $\to$ 32) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(32 $\to$ 16) $\to$ ReLU $\to$ Linear(16 $\to$ 1).
- [x] **Huấn luyện chuẩn mực:** Tối ưu hóa bằng `BCEWithLogitsLoss` với cơ chế **Early Stopping** (`patience = 15`) trên tập Validation.
- [x] **Ánh xạ TabularCPD (Row-major C-order):** Duyệt qua toàn bộ $2 \times 3 \times 2 \times 2 \times 2 \times 2 = 96$ tổ hợp cha bằng `itertools.product`, dự báo xác suất qua mạng MLP, làm mịn $\epsilon$-clipping ($10^{-4}$), và đóng gói thành `TabularCPD`.
- [x] **Tích hợp mạng Bayes:** Sao chép các CPDs của $Z$ và $T$ từ B0, thay thế CPD của $Y$ bằng Neural TabularCPD, gọi `h1_bn.check_model()` thành công.

### 4. Kiểm tra Mô hình H2 - Neural Causal Discovery (Section 10)
- [x] **Tối ưu hóa Gradient liên tục:** Triển khai thuật toán NOTEARS với hàm phạt phi chu trình vi phân $h(W_{\text{eff}}) = \text{Tr}(\exp(W_{\text{eff}} \odot W_{\text{eff}})) - d = 0$.
- [x] **Mặt nạ Cấm cạnh Thời gian (Temporal Forbidden Mask $M$):**
  - Cấm đường chéo chính: $M_{ii} = 0$.
  - Cấm cạnh đi ra từ biến kết quả: $M_{\text{idx}(Y), \ :} = 0$ ($Y \not\to T$, $Y \not\to Z_k$).
  - Cấm cạnh ngược từ can thiệp sang biến nền: $M_{\text{idx}(T), \ \text{idx}(Z_k)} = 0$.
  - Nhân ma trận Hadamard: $W_{\text{eff}} = M \odot W$ ở mọi bước forward.
- [x] **Sửa chu trình (Cycle Repair):** Tự động phát hiện và cắt bỏ cạnh có trọng số tuyệt đối nhỏ nhất trong chu trình nếu xấp xỉ số học còn sót lại.
- [x] **Khớp tham số non-neural:** Khớp bảng CPDs bằng Bayesian smoothing (BDeu prior, ESS=5) tương tự B0.

### 5. Kiểm tra Động cơ Truy vấn Vector hóa $O(1)$ & Hiệu năng
- [x] **Không dùng loop suy luận tuần tự:** Không gọi `VariableElimination.query()` trong vòng lặp từng dòng.
- [x] **Tra cứu Tensor NumPy trực tiếp:** Trích xuất mảng đa chiều `cpd_y.values[tuple(indexer)]` cho toàn bộ $545$ dòng trong 1 lệnh duy nhất.
- [x] **Tốc độ thực thi:** Thời gian tính toán cho 545 dòng kiểm thử đạt **dưới 1 mili-giây** ($0.0008\text{s}$).
- [x] **Tổng runtime 5 Paired Runs:** Toàn bộ 5 runs của B0, H1, H2 hoàn tất trong **dưới 30 giây** trên CPU tiêu chuẩn.

### 6. Kiểm tra Quy chuẩn Đánh giá 5 Paired Runs (Bảng A & Bảng B)
- [x] **Phân lập Causal vs Prediction:**
  - Bảng A đo lường trên nhãn thực tế $Y_{\text{test}}$: AUROC, Macro-F1 (ngưỡng tối ưu trên Val set), Brier Score, Precision, Recall.
  - Bảng B đo lường can thiệp trên tập $C_{\text{test}}$: **Tuyệt đối không dùng $Y_{\text{test}}$**.
- [x] **Tính toán Can thiệp Chuẩn hóa:**
  $$\text{risk}(t) = \frac{1}{N_{\text{test}}} \sum_{i \in \text{TEST}} P(Y=1 \mid do(T=t), C_i)$$
  $$\text{ATE} = \text{risk}(1) - \text{risk}(0)$$
- [x] **Khoảng tin cậy 95% Bootstrap:** Sử dụng 1,000 lượt lấy mẫu lại có hoàn lại trên hiệu ứng cá thể của tập Test.
- [x] **Chênh lệch ghép cặp có dấu:** Tính toán đầy đủ $\Delta_{\text{AUROC vs } B_0}$, $\Delta_{\text{F1 vs } B_0}$, $\Delta_{\text{ATE vs } B_0}$.
- [x] **Xuất bản file:** Đã tạo và lưu thành công `artifacts/tables/Table_A_Prediction.csv` và `artifacts/tables/Table_B_Intervention.csv`.

### 7. Kiểm tra Độ bền vững & Phản bác DoWhy (Bảng C)
- [x] **Placebo Treatment Refuter:** Hoán vị ngẫu nhiên nhãn Treatment $T$. Kết quả hiệu ứng giả triệt tiêu về xấp xỉ 0 ($|\text{ATE}| < 0.05, p > 0.05$), ghi nhận trạng thái **Pass: True**.
- [x] **Data Subset Refuter:** Rút ngẫu nhiên 80% dữ liệu. Hiệu ứng ước lượng lại ổn định so với giá trị ban đầu, ghi nhận trạng thái **Pass: True**.
- [x] **Độ tản mạn ghép cặp (Paired Spread):** Đo lường tính ổn định $\text{Spread} = \max(\text{ATE}) - \min(\text{ATE})$ qua 5 seeds.
- [x] **Xuất bản file:** Đã tạo và lưu thành công `artifacts/tables/Table_C_Robustness_Cost.csv`.

### 8. Kiểm tra Nghiên cứu Bóc tách (Ablations) & Đối chuẩn Causal ML
- [x] **Ablation A1 (H1 Không Dropout):** Đã kiểm chứng dropout giúp giảm Brier Score từ $0.1674$ xuống $0.1667$, ngăn ngừa hiện tượng quá tự tin (overconfidence) trên các ô thưa.
- [x] **Ablation A2 (H2 Không Mask Cấm):** Đã phát hiện và ghi nhận thực nghiệm: khi không có mask $M$, NOTEARS học ra cạnh ngược thời gian $Y \to Z_3$ và tạo chu trình kín làm vỡ mô hình Bayes.
- [x] **Ablation A3 (B0 với MLE):** Đã chứng minh phương pháp MLE thất bại do chia cho 0 tại các ô thưa, làm nổi bật tính bắt buộc của BDeu smoothing.
- [x] **Độ nhạy phân phối tiên nghiệm (ESS Sweep):** Đã kiểm chứng B0 có độ nhạy cực thấp khi quét $ESS \in [1, 20]$ (biến động ATE $< 0.0013$).
- [x] **Hội tụ liên ngành Causal ML:** Đã đối chuẩn với IPW (Hajek: $+0.2580$) và AIPW (Doubly Robust: $+0.2299$). Hiệu ứng của B0 ($+0.2370$) và H1 ($+0.2228$) hội tụ nhất quán trong khoảng này.
- [x] **Xuất bản file mở rộng:** Đã lưu `artifacts/tables/Table_Extended_Ablations.csv`.

### 9. Kiểm tra Tính Toàn vẹn File & Khả năng Tái lập (Reproducibility)
- [x] **Thực thi không lỗi:** Đã chạy kiểm thử tự động toàn bộ 12 code cells trong `causal_modeling_uci697.ipynb` từ đầu đến cuối với mã thoát `exit code 0`.
- [x] **Không phụ thuộc trạng thái toàn cục ẩn:** Mỗi seed khởi tạo lại mầm ngẫu nhiên hoàn toàn độc lập (`torch.manual_seed`, `np.random.seed`).
- [x] **Cơ chế Fallback Console UTF-8:** Đã cấu hình `sys.stdout.reconfigure(encoding='utf-8')` chống lỗi bảng mã `cp1252` trên Windows.
- [x] **Hiển thị đẹp mắt trong Notebook:** Các bảng A, B, C được hiển thị trực tiếp bằng định dạng Pandas DataFrame chuẩn hóa.

---

## III. KẾT LUẬN NGHIỆM THU

| Hạng mục kiểm tra | Số tiêu chí | Số tiêu chí Đạt | Tỷ lệ Đạt | Kết luận |
| :--- | :---: | :---: | :---: | :---: |
| 1. Tiền xử lý & Splits | 5 | 5 | 100% | **APPROVED** |
| 2. Mô hình B0 (Baseline) | 5 | 5 | 100% | **APPROVED** |
| 3. Mô hình H1 (Hybrid Neural) | 5 | 5 | 100% | **APPROVED** |
| 4. Mô hình H2 (Neural Discovery) | 4 | 4 | 100% | **APPROVED** |
| 5. Động cơ Vector hóa O(1) | 4 | 4 | 100% | **APPROVED** |
| 6. Đánh giá 5 Paired Runs (Bảng A, B) | 5 | 5 | 100% | **APPROVED** |
| 7. DoWhy Refutations (Bảng C) | 4 | 4 | 100% | **APPROVED** |
| 8. Nghiên cứu Bóc tách & Causal Baselines | 6 | 6 | 100% | **APPROVED** |
| 9. Tính Toàn vẹn & Tái lập | 4 | 4 | 100% | **APPROVED** |
| **TỔNG CỘNG** | **42** | **42** | **100%** | **CHÍNH THỨC THÔNG QUA (PASSED)** |

> **Chữ ký xác nhận nghiệm thu:** Nhóm nghiên cứu Causal ML — Dự án IT701100  
> **Trạng thái:** Toàn bộ tiêu chí đạt 100%, sẵn sàng cho Báo cáo Midterm và Báo cáo Final.
