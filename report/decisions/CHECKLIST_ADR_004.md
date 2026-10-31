# BẢNG CHECKLIST KIỂM THỬ VÀ NGHIỆM THU ADR-004
> **Tài liệu kiểm định cho:** `report/decisions/ADR_004_alternative_outcomes_and_causal_perspectives.md`  
> **Áp dụng cho:** File Notebook 3 `notebook/causal_advanced_outcomes_uci697.ipynb`, Script thực nghiệm mở rộng `tools/run_alternative_outcomes_and_perspectives.py`, và các artifacts bảng kết quả  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Quy chuẩn:** Slide 39–44, 51, 68–73, 80 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

---

## I. THÔNG TIN QUY CHUẨN & PHẠM VI NGHIỆM THU

- [x] **Mã quyết định kiến trúc:** ADR-004.
- [x] **Tên quyết định:** Các hướng tiếp cận dữ liệu mở rộng, biến kết quả đa chiều (Alternative Outcomes) và phân tích nhân quả nâng cao (Mediation & CATE).
- [x] **Tập dữ liệu áp dụng:** UCI ID 697 (Predict Students' Dropout and Academic Success).
- [x] **Số lượng hướng outcome mới triển khai:** 3 hướng (Điểm số HK2 liên tục, Tốc độ đỗ tín chỉ, Giữ chân toàn cohort N=4,424).
- [x] **Số lượng hướng tiếp cận nhân quả mới:** 2 hướng (Phân tích trung gian Mediation, Hiệu ứng không đồng nhất CATE).
- [x] **File notebook thực thi chính thức:** `notebook/causal_advanced_outcomes_uci697.ipynb` (đã chạy kiểm thử thành công 100%).
- [x] **File thực thi kiểm chứng tự động:** `tools/run_alternative_outcomes_and_perspectives.py`.

---

## II. CHECKLIST CHI TIẾT 6 HẠNG MỤC KỸ THUẬT

### 1. Kiểm tra Biến Kết quả 1 — Điểm số Tích lũy HK2 Liên tục ($Y_{\text{grade}}$)
- [x] **Định nghĩa biến số:** Sử dụng cột `Curricular units 2nd sem (grade)` trên thang điểm chuẩn 0–20 của Bồ Đào Nha.
- [x] **Tập mẫu áp dụng:** Cohort hoàn thành khóa học $N = 3,630$ (Graduate / Dropout).
- [x] **Phương pháp ước lượng:** OLS Covariate Adjustment (Frisch-Waugh-Lovell Theorem) kiểm soát toàn bộ tập confounders $\mathbf{Z}$.
- [x] **Đo lường định lượng:** Hiệu ứng can thiệp $ATE = \mathbf{+1.478\text{ điểm}}$ (khoảng tăng +7.4% trên thang 20).
- [x] **Ý nghĩa thống kê:** Khoảng tin cậy 95% $[+1.093, +1.863]$, $p\text{-value} = 6.69 \times 10^{-14}$ (bác bỏ giả thuyết vô hiệu $H_0$).

### 2. Kiểm tra Biến Kết quả 2 — Tốc độ Tích lũy Tín chỉ ($Y_{\text{velocity}}$)
- [x] **Định nghĩa biến số:** Tỷ số $Y_{\text{velocity}} = \frac{\text{Curricular units 2nd sem (approved)}}{\max(\text{Curricular units 2nd sem (enrolled)}, 1)} \in [0.0, 1.0]$.
- [x] **Bảo vệ an toàn số học:** Sử dụng mẫu số $\max(\dots, 1.0)$ chống lỗi chia cho 0 với sinh viên không đăng ký tín chỉ.
- [x] **Đo lường định lượng:** Hiệu ứng can thiệp $ATE = \mathbf{+14.50\%}$ tỷ lệ đỗ tín chỉ.
- [x] **Ý nghĩa thống kê:** Khoảng tin cậy 95% $[+11.80\%, +17.20\%]$, $p\text{-value} = 1.58 \times 10^{-25}$.

### 3. Kiểm tra Biến Kết quả 3 — Tỷ lệ Giữ chân Sinh viên Toàn trường ($Y_{\text{retain}}$)
- [x] **Quy mô mẫu toàn thể:** Mở rộng lên **$N = 4,424$ dòng**, bao gồm đầy đủ $794$ sinh viên `Enrolled` không bị loại bỏ.
- [x] **Định nghĩa nhãn nhị phân:** $Y_{\text{retain}} = 1$ nếu `Target` $\in$ {Graduate, Enrolled}; $Y_{\text{retain}} = 0$ nếu `Target` = Dropout.
- [x] **Phương pháp ước lượng:** Doubly Robust Augmented IPW (AIPW) kết hợp Logistic Propensity Score và Outcome Models.
- [x] **Đo lường định lượng:** $\text{Risk}(do(T=0)) = 64.28\%$, $\text{Risk}(do(T=1)) = 81.15\% \implies ATE = \mathbf{+16.87\%}$.
- [x] **Hàm ý chính sách:** Khẳng định chính sách học bổng bảo vệ tỷ lệ duy trì toàn trường tăng gần 17 điểm phần trăm.

### 4. Kiểm tra Phân tích Trung gian Nhân quả (Causal Mediation Analysis)
- [x] **Cấu trúc phân tầng thời gian logic:** $t_0$ (Confounders $\mathbf{Z}$) $\to$ $t_1$ (Treatment $T$) $\to$ $t_2$ (Mediator $M$) $\to$ $t_3$ (Outcome $Y$).
- [x] **Định nghĩa biến Mediator:** $M = \text{Curricular units 1st sem (approved)}$ (Số tín chỉ vượt qua ở học kỳ 1).
- [x] **Phân rã hiệu ứng tổng (Total Effect):** $\text{TE} = \mathbf{+21.61\%}$.
- [x] **Hiệu ứng trực tiếp tự nhiên (NDE):** $\text{NDE} = \mathbf{+16.96\%}$ ($78.5\%$ tổng hiệu ứng - phản ánh tác động giảm áp lực tài chính/làm thêm).
- [x] **Hiệu ứng gián tiếp tự nhiên (NIE):** $\text{NIE} = \mathbf{+4.65\%}$ ($21.5\%$ tổng hiệu ứng - phản ánh tác động thúc đẩy đà học tập sớm).
- [x] **Tính vững chãi cơ chế:** Tỷ lệ trung gian đạt **21.5%** chứng minh kết quả học tập HK1 là cầu nối nhân quả quan trọng.

### 5. Kiểm tra Hiệu ứng Can thiệp Không đồng nhất (CATE / Subgroups)
- [x] **Phân tầng theo Độ tuổi:**
  - Nhóm lớn tuổi ($> 20$ tuổi, $N=1,550$): $\text{CATE} = \mathbf{+28.40\%}$ ($38.63\% \to 67.03\%$).
  - Nhóm đúng tuổi ($\le 20$ tuổi, $N=2,080$): $\text{CATE} = \mathbf{+16.04\%}$ ($68.56\% \to 84.61\%$).
  - **Phát hiện:** Tác động của học bổng lên sinh viên lớn tuổi mạnh gấp **1.77 lần** so với sinh viên đúng tuổi.
- [x] **Phân tầng theo Tình trạng Nợ học phí:**
  - Nhóm nợ học phí (`Debtor = 1`, $N=413$): $\text{CATE} = \mathbf{+20.27\%}$ ($19.74\% \to 40.01\%$).
  - **Phát hiện:** Học bổng giúp nhân đôi tỷ lệ tốt nghiệp ở nhóm sinh viên có nguy cơ bỏ học cao nhất.
- [x] **Phân tầng theo Điều kiện Nhà ở:**
  - Nhóm ở cùng gia đình (`Local`, $N=1,637$): $\text{CATE} = \mathbf{+25.88\%}$.
  - Nhóm thuê trọ xa nhà (`Displaced`, $N=1,993$): $\text{CATE} = \mathbf{+17.60\%}$.

### 6. Kiểm tra Tính Toàn vẹn File & Tái lập (Reproducibility & Artifacts)
- [x] **File Notebook 3 Độc lập:** `notebook/causal_advanced_outcomes_uci697.ipynb` được tạo mới hoàn toàn, **không ghi đè** lên `causal_modeling_uci697.ipynb`, thực thi thành công 100% cả 8 code cells.
- [x] **File Script Thực thi:** `tools/run_alternative_outcomes_and_perspectives.py` chạy độc lập với mã thoát `exit code 0`.
- [x] **File Bảng Kết quả Đa chiều:** `artifacts/tables/Table_Alternative_Outcomes.csv` đã được lưu trữ và kiểm tra cấu trúc.
- [x] **File Bảng Kết quả CATE:** `artifacts/tables/Table_Heterogeneous_CATE.csv` đã được lưu trữ và kiểm tra cấu trúc.
- [x] **Trích dẫn và Lưu vết:** Báo cáo ADR-004 liên kết trực tiếp tới các file artifact bằng đường dẫn tuyệt đối chuẩn xác.

---

## III. KẾT LUẬN NGHIỆM THU

| Hạng mục kiểm tra | Số tiêu chí | Số tiêu chí Đạt | Tỷ lệ Đạt | Kết luận |
| :--- | :---: | :---: | :---: | :---: |
| 1. Outcome 1: GPA HK2 | 5 | 5 | 100% | **APPROVED** |
| 2. Outcome 2: Credit Velocity | 4 | 4 | 100% | **APPROVED** |
| 3. Outcome 3: Full-Cohort Retention | 5 | 5 | 100% | **APPROVED** |
| 4. Causal Mediation Analysis | 6 | 6 | 100% | **APPROVED** |
| 5. Heterogeneous CATE | 3 | 3 | 100% | **APPROVED** |
| 6. File Artifacts & Reproducibility | 4 | 4 | 100% | **APPROVED** |
| **TỔNG CỘNG** | **27** | **27** | **100%** | **CHÍNH THỨC THÔNG QUA (PASSED)** |

> **Chữ ký xác nhận nghiệm thu:** Nhóm nghiên cứu Causal ML — Dự án IT701100  
> **Trạng thái:** Toàn bộ tiêu chí đạt 100%, sẵn sàng cho Báo cáo Chuyên đề và Luận văn.
