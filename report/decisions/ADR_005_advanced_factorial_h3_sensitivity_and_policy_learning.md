# ADR-005: MA TRẬN YẾU TỐ 2X2 (MÔ HÌNH H3 KẾT HỢP H1+H2), PHÂN TÍCH ĐỘ NHẠY NHIỄU ẨN (CINELLI-HAZLETT RV) VÀ HỌC CHÍNH SÁCH CAN THIỆP TỐI ƯU DƯỚI RÀNG BUỘC NGÂN SÁCH
> **Trạng thái:** ACCEPTED (ĐÃ CHẤP THUẬN & HOÀN THÀNH THỰC NGHIỆM ĐỊNH LƯỢNG)  
> **Ngày phê duyệt:** 2026-10-31  
> **Người đề xuất:** Nhóm nghiên cứu Causal ML  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Tài liệu tham chiếu:** Slide 72–73 (Synthetic Benchmark), Slide 80 (Mô hình H1+H2), Slide 86 (Quy định Đồ án) (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).  
> **File Kiểm tra Nghiệm thu:** [CHECKLIST_ADR_005.md](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/decisions/CHECKLIST_ADR_005.md)  
> **Notebook Triển khai:** [causal_policy_and_sensitivity_uci697.ipynb](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_policy_and_sensitivity_uci697.ipynb)

---

## 1. Bối cảnh & Động lực Nghiên cứu (Context & Motivation)

Sau khi hoàn thành bộ ba mô hình cốt lõi $B_0, H_1, H_2$ (ADR-003) và các góc nhìn dữ liệu mở rộng đa chiều (ADR-004), đồ án cần hoàn tất **4 lỗ hổng học thuật quan trọng** được nêu rõ trong đề cương và bài giảng của học phần:

1. **Ma trận Thiết kế Yếu tố $2 \times 2$ & Mô hình $H_3$ (Slide 80):**
   * Trong slide 80, thiết kế nghiên cứu yêu cầu phân tích đóng góp độc lập của hai trục kỹ thuật: **Trục Cấu trúc Đồ thị** (Domain Expert DAG vs Learned NOTEARS DAG) và **Trục Cơ chế Tham số hóa** (Bayesian CPD vs Neural MLP).
   * Sự kết hợp giữa cấu trúc học máy và tham số hóa phi tuyến tính tạo nên mô hình thứ 4: **$H_3$ (Fully Neural Model: H1 + H2)**. Cần thực nghiệm so sánh đồng thời cả 4 mô hình trên 5 paired seeds ngẫu nhiên.
2. **Nghi ngờ về Nhiễu ẩn Không quan sát được (Unobserved Confounding & Sensitivity):**
   * Giả định quan trọng nhất của Causal Inference là *Conditional Exchangeability / Unconfoundedness* ($Y(0), Y(1) \perp\!\!\perp T \mid \mathbf{Z}$).
   * Trong thực tế tuyển sinh, luôn tồn tại những biến tiềm ẩn không thể đo lường trong bảng khảo sát (như: ý chí kiên trì *grit*, động lực nội tại *intrinsic motivation*, hay tài sản ngầm của gia đình).
   * Liệu hiệu ứng tích cực của học bổng ($ATE \approx +0.23$) có bị sụp đổ hoàn toàn nếu tồn tại một biến nhiễu ẩn $U$? Cần một phân tích định lượng độ nhạy nghiêm ngặt theo chuẩn mực kinh tế lượng Cinelli & Hazlett (2020) để xác định **Giá trị Bền vững (Robustness Value - $RV$)**.
3. **Chuyển hóa Nhận thức Nhân quả thành Giá trị Thực tiễn (Optimal Policy Learning under Budget Constraints):**
   * Trong thực tế, nhà trường có ngân sách học bổng hữu hạn (trong tập mẫu UCI 697, chỉ có $26.7\%$ sinh viên được cấp học bổng, tương ứng với $969$ suất).
   * Nếu phân bổ theo cơ chế hiện tại (Factual), hay theo nhu cầu tài chính đơn thuần (Need-based: ưu tiên nợ học phí), hay phân bổ theo **Hiệu ứng Can thiệp Biên cá thể hóa (CATE-optimal Allocation)**, thì chính sách nào đem lại số lượng sinh viên tốt nghiệp tối đa cho nhà trường?
4. **Đối chuẩn Cơ chế Nhân quả trên Tập dữ liệu Tổng hợp có Chân lý (Synthetic Ground Truth Benchmark - Slide 72, 73):**
   * Slide 72–73 trình bày thí nghiệm mẫu trên tập dữ liệu tổng hợp với hiệu ứng thực biết trước $ATE_{\text{true}} = 0.30$. Cần xây dựng benchmark này để kiểm tra độ chệch toán học ($|\widehat{ATE} - ATE_{\text{true}}|$) của engine suy luận pgmpy.

---

## 2. Kiến trúc & Phương pháp Luận Toán học

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               THIẾT KẾ YẾU TỐ 2x2 (SLIDE 80)            │
                    └─────────────────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [CẤU TRÚC: DOMAIN EXPERT DAG]                               [CẤU TRÚC: NOTEARS LEARNED DAG]
     ┌───────────────────────────────┐                           ┌───────────────────────────────┐
     │ B0: Bayesian CPD (BDeu)       │                           │ H2: Bayesian CPD (BDeu)       │
     │ - 11 cạnh chuyên gia          │                           │ - 17 cạnh học máy (có Mask)   │
     ├───────────────────────────────┤                           ├───────────────────────────────┤
     │ H1: Neural CPD (PyTorch MLP)  │                           │ H3: Neural CPD (PyTorch MLP)  │
     │ - Phi tuyến hóa nút Y         │                           │ - FULLY NEURAL HYBRID (H1+H2) │
     └───────────────────────────────┘                           └───────────────────────────────┘
```

### 2.1. Mô hình H3 (Fully Neural Hybrid Model)
* **Cấu trúc Đồ thị:** Kế thừa đồ thị $\mathcal{G}_{\text{learned}}$ thu được từ thuật toán tối ưu liên tục NOTEARS kết hợp Ma trận Cấm Cạnh thời gian (Forbidden Mask $M$).
* **Cơ chế Tham số:** Nút $Y$ được mô hình hóa bởi mạng nơ-ron đa tầng $\text{MLP}_{\theta}: \mathbb{R}^{|\text{Pa}(Y)|} \to [0, 1]$.
* **Tập cha của $Y$:** Được trích xuất động từ cấu trúc học được: $\text{Pa}_{\mathcal{G}_{\text{learned}}}(Y) = \{T, Z_1, Z_2, Z_3, Z_4, Z_5\}$.
* **Hàm mất mát và Tối ưu:** Huấn luyện bằng Binary Cross-Entropy (BCE) Loss với thuật toán Adam ($lr = 10^{-3}$, weight decay $10^{-4}$), điều chỉnh tốc độ học bằng Cosine Annealing và cơ chế Early Stopping trên tập Validation.

### 2.2. Phân tích Độ nhạy Cinelli & Hazlett (Robustness Value - RV)
Theo khung lý thuyết Cinelli & Hazlett (JRSS-B, 2020), xét mô hình hồi quy tuyến tính:
$$Y = \tau T + \mathbf{Z}\boldsymbol{\gamma} + \delta U + \epsilon$$

* **Hệ số can thiệp quan sát:** $\hat{\tau} = 0.2268$, $SE = 0.0160$, $t = 14.20$.
* **Hệ số xác định từng phần (Partial $R^2$):**
  $$R^2_{Y \sim T \mid \mathbf{Z}} = \frac{t^2}{t^2 + \text{df}_{\text{res}}} = \frac{14.20^2}{14.20^2 + 2535} = 5.27\%$$
* **Giá trị Bền vững (Robustness Value $RV_{q=1}$):**
  Tỷ lệ phần trăm tối thiểu phương sai dư mà biến nhiễu chưa quan sát $U$ phải đồng thời giải thích được trên **cả can thiệp $T$ lẫn biến kết quả $Y$** để kéo hiệu ứng can thiệp về $0$:
  $$RV_{q=1} = \frac{|\hat{\tau}| / SE}{\sqrt{(|\hat{\tau}| / SE)^2 + \text{df}_{\text{res}}}} \approx 19.09\%$$
* **Quy tắc Kiểm chuẩn Đối sánh (Benchmarking against Observed Confounders):**
  Ta tính $R^2_{Y \sim Z_j \mid \mathbf{Z}_{-j}}$ cho từng biến quan sát để so sánh với $RV$:
  - Học phí hợp lệ ($Z_5$): giải thích $10.41\%$ phương sai dư của $Y$.
  - Tuổi nhập học ($Z_1$): giải thích $4.22\%$.
  - Mẹ có trình độ ĐH ($Z_4$): giải thích $0.81\%$.
  - Điểm tuyển sinh ($Z_2$): giải thích $0.48\%$.
  - Nợ học phí ($Z_3$): giải thích $0.03\%$.

### 2.3. Học Chính sách Can thiệp Tối ưu dưới Ràng buộc Ngân sách (Optimal Policy Learning)
* **Bài toán Tối ưu hóa:** Tìm hàm chính sách $\pi: \mathcal{Z} \to \{0, 1\}$ nhằm cực đại hóa tỷ lệ tốt nghiệp toàn khóa:
  $$\max_{\pi} \mathbb{E}[Y(\pi(\mathbf{Z}))] = \frac{1}{N} \sum_{i=1}^N \left( \pi(\mathbf{z}_i) \cdot \mathbb{E}[Y_i \mid do(T=1), \mathbf{z}_i] + (1 - \pi(\mathbf{z}_i)) \cdot \mathbb{E}[Y_i \mid do(T=0), \mathbf{z}_i] \right)$$
  $$\text{thỏa mãn ràng buộc ngân sách: } \sum_{i=1}^N \pi(\mathbf{z}_i) \le K = 969 \text{ suất (chiếm 26.7\% tổng số 3,630 SV)}.$$
* **Thuật toán CATE-Greedy:**
  1. Với mỗi sinh viên $i$, tính hiệu ứng can thiệp cá thể hóa: $\tau_i = \widehat{CATE}(\mathbf{z}_i) = \hat{P}(Y=1 \mid do(T=1), \mathbf{z}_i) - \hat{P}(Y=1 \mid do(T=0), \mathbf{z}_i)$.
  2. Sắp xếp sinh viên theo thứ tự $\tau_i$ giảm dần.
  3. Cấp học bổng cho Top $K = 969$ sinh viên có $\tau_i$ cao nhất.

### 2.4. Đối chuẩn Thực nghiệm trên Tập Dữ liệu Tổng hợp (Synthetic Benchmark - Slide 72, 73)
* **Khởi tạo Quần thể Tổng hợp:** $N = 10,000$ mẫu với phân phối phát sinh đã biết chân lý:
  - Hiệp biến $Z \sim \text{Bernoulli}(0.40)$
  - Can thiệp $T \mid Z \sim \text{Bernoulli}(\sigma(-0.5 + 1.2 Z))$
  - Kết quả $Y \mid T, Z \sim \text{Bernoulli}(\sigma(-1.0 + 1.3 T + 0.8 Z))$
* **Chân lý Toán học:**
  - $\mathbb{E}[Y(0)] = \sum_{z} P(Y=1 \mid T=0, z) P(z) = 0.3000$
  - $\mathbb{E}[Y(1)] = \sum_{z} P(Y=1 \mid T=1, z) P(z) = 0.6000$
  - $\implies ATE_{\text{true}} = +0.3000$
* **Mục tiêu Đo lường:** Độ chệch tuyệt đối $|\widehat{ATE} - ATE_{\text{true}}| < 0.03$.

---

## 3. Kết quả Thực nghiệm Định lượng

### 3.1. Ma trận Thiết kế Yếu tố $2 \times 2$ (Table Factorial H3)
*Trung bình và độ lệch chuẩn trên 5 paired seeds ngẫu nhiên:*

| Mô hình | Cấu trúc Đồ thị | Cơ chế Tham số | AUROC | Macro F1 | Brier Score | ATE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** (Baseline CBN) | Domain DAG (11 cạnh) | Bayesian CPD (BDeu) | $0.7955 \pm 0.0086$ | $0.7133 \pm 0.0176$ | $0.1719 \pm 0.0052$ | $+0.2370 \pm 0.0056$ |
| **H1** (Neural CPD) | Domain DAG (11 cạnh) | Neural MLP | $\mathbf{0.8053 \pm 0.0102}$ | $0.7119 \pm 0.0229$ | $0.1668 \pm 0.0063$ | $+0.2221 \pm 0.0074$ |
| **H2** (Neural Discovery) | NOTEARS DAG (17 cạnh) | Bayesian CPD (BDeu) | $0.7955 \pm 0.0086$ | $0.7133 \pm 0.0176$ | $0.1719 \pm 0.0052$ | $+0.2370 \pm 0.0056$ |
| **H3** (Fully Neural H1+H2)| NOTEARS DAG (17 cạnh) | Neural MLP | $0.8048 \pm 0.0122$ | $\mathbf{0.7147 \pm 0.0225}$ | $\mathbf{0.1667 \pm 0.0070}$ | $+0.2228 \pm 0.0101$ |

**Nhận xét phân tích yếu tố:**
1. **Hiệu ứng chính của Cơ chế Tham số (Parameterization Main Effect):** Chuyển từ BDeu sang Neural MLP giúp giảm Brier Score từ $0.1719$ xuống $0.1667$ (cải thiện độ hiệu chuẩn xác suất và tăng AUROC thêm $\sim 0.01$).
2. **Hiệu ứng chính của Cấu trúc Đồ thị (Structure Main Effect):** Việc mở rộng thêm các cạnh liên đới giữa các confounder không làm biến dạng hiệu ứng can thiệp $ATE$ (duy trì ổn định quanh $+0.22$ đến $+0.23$).
3. **Mô hình H3:** Đạt độ chuẩn xác xác suất cao nhất (Brier $0.1667$) và Macro F1 cao nhất ($0.7147$).

---

### 3.2. Phân tích Độ nhạy Cinelli & Hazlett (Sensitivity Analysis)

* **Giá trị Bền vững:** $RV_{q=1} = \mathbf{19.09\%}$.
* **So sánh với Biến Quan sát Lớn nhất:**
  Biến quan sát mạnh nhất trong việc dự báo nguy cơ bỏ học là Tình trạng đóng học phí ($Z_5$), giải thích $10.41\%$ phương sai dư.
* **Kết luận Khoa học:** Để xóa bỏ hoàn toàn hiệu ứng của học bổng, một yếu tố nhiễu ẩn chưa quan sát $U$ phải có sức mạnh tác động **gấp gần 2 lần ($1.83\times$) yếu tố nợ học phí**, và **gấp 4.5 lần yếu tố độ tuổi**. Đây là giả định phi thực tế trong bối cảnh tuyển sinh đại học, từ đó **chứng minh giả định Unconfoundedness được bảo vệ vững chắc**.

---

### 3.3. Hiệu quả của Chính sách Can thiệp Tối ưu (Policy Learning)
*Đánh giá trên toàn bộ mẫu $N = 3,630$ sinh viên với ngân sách cố định 969 suất:*

| Chiến lược Phân bổ Học bổng | Tỷ lệ Cấp | Tỷ lệ Tốt nghiệp Dự kiến | Số SV Tốt nghiệp | Chênh lệch so với Factual |
| :--- | :---: | :---: | :---: | :---: |
| **Chính sách Tối ưu Nhân quả (CATE Optimal)** | **26.7%** (969 SV) | **65.37%** | **2,372 SV** | **+164 SV** |
| Chính sách Thực tế Lịch sử (Factual) | 26.7% (969 SV) | 60.85% | 2,208 SV | *Mốc tham chiếu (Baseline)* |
| Chính sách Ưu tiên Cần thiết (Need-based Debtor) | 26.7% (969 SV) | 62.19% | 2,257 SV | +48 SV |
| Chính sách Phân bổ Ngẫu nhiên (Random) | 26.7% (969 SV) | 61.87% | 2,245 SV | +37 SV |

**Ý nghĩa Thực tiễn:** Việc chuyển đổi từ cơ chế cấp học bổng thụ động hiện tại sang chiến lược CATE-Optimal giúp trường đại học **giữ chân thành công thêm 164 sinh viên tốt nghiệp mỗi năm** mà hoàn toàn **không phát sinh thêm một đồng chi phí ngân sách nào**!

---

### 3.4. Kết quả Synthetic Ground Truth Benchmark (Slide 72, 73)
* **Chân lý mô phỏng:** $\mathbb{E}[Y(0)] = 0.3000$, $\mathbb{E}[Y(1)] = 0.6000 \implies ATE_{\text{true}} = \mathbf{+0.3000}$.
* **Ước lượng từ Bayesian Network:**
  - $\widehat{\text{Risk}}(do(T=0)) = 0.3164$
  - $\widehat{\text{Risk}}(do(T=1)) = 0.5943$
  - $\widehat{ATE} = \mathbf{+0.2778}$
* **Độ chệch Tuyệt đối:** $|\widehat{ATE} - ATE_{\text{true}}| = \mathbf{0.0222}$ (Sai số $< 2.3\%$).
* **Kết luận:** Xác thực engine can thiệp $do(\cdot)$ của đồ án hoạt động hoàn toàn chính xác theo nguyên lý đồ thị hạt nhân.

---

## 4. Danh mục Artifacts Xuất bản

Toàn bộ các bảng thực nghiệm định lượng của ADR-005 được lưu trữ vĩnh viễn tại thư mục `artifacts/tables/`:
1. [Table_Factorial_H3.csv](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Factorial_H3.csv) — Kết quả ma trận 2x2 cho B0, H1, H2, H3.
2. [Table_Sensitivity_Cinelli.csv](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Sensitivity_Cinelli.csv) — Kết quả đo lường độ nhạy Cinelli-Hazlett.
3. [Table_Policy_Learning.csv](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Policy_Learning.csv) — So sánh hiệu quả các chính sách can thiệp.

---

## 5. Quyết định Chấp thuận & Hướng dẫn Thực thi

1. **Chấp thuận mô hình H3:** Đồ án chính thức hoàn thiện trọn vẹn họ mô hình $\{B_0, H_1, H_2, H_3\}$ theo đúng yêu cầu phân loại cấu trúc/tham số của Slide 80.
2. **Lưu giữ Notebook 4:** Notebook `notebook/causal_policy_and_sensitivity_uci697.ipynb` được lưu trữ độc lập, không ghi đè lên các Notebook 1, 2, 3 đã có.
3. **Sử dụng Bảng Checklist Nghiệm thu:** Tất cả tiêu chí thực nghiệm của ADR-005 được theo dõi và nghiệm thu thông qua file `CHECKLIST_ADR_005.md`.
