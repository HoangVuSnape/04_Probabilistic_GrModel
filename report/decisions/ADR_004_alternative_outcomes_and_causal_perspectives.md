# ADR-004: CÁC HƯỚNG TIẾP CẬN DỮ LIỆU MỞ RỘNG, BIẾN KẾT QUẢ ĐA CHIỀU (ALTERNATIVE OUTCOMES) VÀ PHÂN TÍCH NHÂN QUẢ NÂNG CAO (MEDIATION & CATE)
> **Trạng thái:** ACCEPTED (ĐÃ CHẤP THUẬN & HOÀN THÀNH THỰC NGHIỆM ĐỊNH LƯỢNG)  
> **Ngày phê duyệt:** 2026-10-30  
> **Người đề xuất:** Nhóm nghiên cứu Causal ML  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Tài liệu tham chiếu:** Slide 39–44, 51, 68–73, 80 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

---

## 1. Bối cảnh & Động lực Nghiên cứu (Context & Motivation)

Trong các quyết định trước (ADR-002, ADR-003), đồ án đã giải quyết hoàn hảo bài toán cốt lõi:
* Can thiệp nhị phân: $T = \text{Scholarship holder} \in \{0, 1\}$.
* Biến kết quả nhị phân: $Y = \text{Target} \in \{\text{Graduate: 1}, \text{Dropout: 0}\}$, đã lọc bỏ $794$ sinh viên `Enrolled` để đảm bảo bài toán phân loại nhị phân không bị mập mờ nhãn.

Tuy nhiên, từ góc nhìn khoa học dữ liệu và hoạch định chính sách giáo dục đại học (*Higher Education Policy*), cách tiếp cận nhị phân truyền thống bộc lộ **3 giới hạn nhận thức**:
1. **Thiên kiến phân cực (Binarization Loss):** Nhãn $Y \in \{0, 1\}$ chỉ cho biết sinh viên có tốt nghiệp hay không, nhưng hoàn toàn bỏ qua **chất lượng học thuật thực chất** (sinh viên tốt nghiệp với điểm số bao nhiêu, học bổng có giúp sinh viên đạt học lực xuất sắc hay chỉ vừa đủ điểm qua môn?).
2. **Bỏ qua động thái trung gian (Black-box Causal Mechanism):** Con số $ATE \approx +0.23$ chỉ ra rằng học bổng làm tăng tỷ lệ tốt nghiệp, nhưng không trả lời được cơ chế vận hành: *Học bổng tác động trực tiếp do gánh nặng tài chính được giải tỏa ($T \to Y$) hay gián tiếp thông qua việc giúp sinh viên tích lũy đà học tập sớm ở học kỳ 1 ($T \to M \to Y$)?*
3. **Ảo tưởng về hiệu ứng đồng nhất (Homogeneity Fallacy):** Hiệu ứng trung bình toàn trường ($ATE$) giả định mọi sinh viên đều hưởng lợi như nhau từ học bổng. Nhưng trên thực tế, *ai là người hưởng lợi nhiều nhất*? Sinh viên lớn tuổi hay trẻ tuổi? Sinh viên đang mắc nợ học phí hay sinh viên bình thường? Cần ước lượng hiệu ứng can thiệp có điều kiện (**CATE / Heterogeneous Treatment Effects**).

---

## 2. Các Quyết định Kiến trúc & Đề xuất Hướng tiếp cận Mới

```
                                  [CÁC HƯỚNG MỞ RỘNG ADR-004]
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
   [HƯỚNG BIẾN KẾT QUẢ ĐA CHIỀU]                                       [HƯỚNG TIẾP CẬN DỮ LIỆU MỚI]
   1. Outcome 1: GPA / Grade HK2 (Liên tục 0-20)                       1. Causal Mediation Analysis (T -> M -> Y)
   2. Outcome 2: Credit Velocity (Tỷ lệ đỗ tín chỉ)                    2. Heterogeneous Effects (CATE theo Phân tầng)
   3. Outcome 3: Full-Cohort Retention (N=4,424)                       3. Mở rộng Tập Hiệp biến Confounders
```

### 2.1. Nhóm Hướng Biến Kết quả Mới (Alternative Outcomes)

#### Hướng 1: Kết quả Năng lực Học thuật Thực chất Liên tục ($Y_{\text{grade}}$)
* **Định nghĩa biến:** Điểm trung bình môn tích lũy học kỳ 2: $Y_{\text{grade}} = \text{Curricular units 2nd sem (grade)} \in [0, 20]$.
* **Câu hỏi nhân quả:** *"Việc cấp học bổng tác động làm thay đổi điểm số học thuật cuối năm 1 bao nhiêu điểm trên thang 20 sau khi đã triệt tiêu toàn bộ sai lệch đầu vào?"*
* **Phương pháp ước lượng:** Frisch-Waugh-Lovell Covariate Adjustment & IPW liên tục:
  $$\tau_{\text{grade}} = \mathbb{E}[Y_{\text{grade}} \mid do(T=1)] - \mathbb{E}[Y_{\text{grade}} \mid do(T=0)]$$

#### Hướng 2: Tốc độ Hoàn thành Tín chỉ / Hiệu suất Học tập ($Y_{\text{velocity}}$)
* **Định nghĩa biến:** Tỷ lệ tín chỉ đỗ trên tổng tín chỉ đăng ký học kỳ 2:
  $$Y_{\text{velocity}} = \frac{\text{Curricular units 2nd sem (approved)}}{\max(\text{Curricular units 2nd sem (enrolled)}, 1)} \in [0.0, 1.0]$$
* **Ý nghĩa:** Đo lường độ trễ tiến độ (academic pacing / momentum). Sinh viên có học bổng có hoàn thành môn học đúng tiến độ hay bị nợ đọng tín chỉ?

#### Hướng 3: Tỷ lệ Giữ chân Sinh viên Toàn trường Toàn bộ Cohort ($Y_{\text{retain}}, N = 4,424$)
* **Định nghĩa biến:** Không lọc bỏ 794 sinh viên `Enrolled`, mà gom nhóm thành bài toán chống bỏ học toàn trường:
  $$Y_{\text{retain}} = \begin{cases} 1 & \text{nếu Target} \in \{\text{'Graduate'}, \text{'Enrolled'}\} \\ 0 & \text{nếu Target} = \text{'Dropout'} \end{cases}$$
* **Ý nghĩa:** Trả lời trực tiếp bài toán của Ban giám hiệu: *"Học bổng có giúp bảo vệ tỷ lệ duy trì học tập toàn trường hay không, tính trên 100% sinh viên nhập học?"*.

---

### 2.2. Nhóm Hướng Tiếp cận Dữ liệu Mới (Alternative Data Perspectives)

#### Hướng 4: Phân tích Trung gian Nhân quả (Causal Mediation Analysis $T \to M \to Y$)
* **Cấu trúc đồ thị thời gian 3 pha:**
  * $t_0$: Hiệp biến nền $\mathbf{Z}$ (Tuổi, điểm tuyển sinh, nợ học phí, mẹ, trọ).
  * $t_1$: Can thiệp trao học bổng $T$.
  * $t_2$: Biến trung gian $M = \text{Curricular units 1st sem (approved)}$ (Số tín chỉ đỗ học kỳ 1).
  * $t_3$: Biến kết quả $Y = \text{Tốt nghiệp (1) / Bỏ học (0)}$.
* **Phân rã hiệu ứng tổng (Total Effect Decomposition):**
  $$\text{Total Effect (TE)} = \text{Natural Direct Effect (NDE)} + \text{Natural Indirect Effect (NIE)}$$
  * **NDE ($T \to Y$ trực tiếp):** Tác động tâm lý an tâm tài chính, giảm áp lực làm thêm ngoài giờ.
  * **NIE ($T \to M \to Y$ gián tiếp):** Tác động thúc đẩy đà học tập: học bổng giúp đỗ nhiều môn ở HK1, từ đó tạo nền tảng để tốt nghiệp.

#### Hướng 5: Hiệu ứng Can thiệp Không đồng nhất (Heterogeneous Treatment Effects - CATE)
* Đo lường hiệu ứng phân tầng theo từng phân khúc sinh viên:
  $$\tau(x) = \mathbb{E}[Y \mid do(T=1), X=x] - \mathbb{E}[Y \mid do(T=0), X=x]$$
* Các chiều phân tầng mục tiêu:
  1. *Tình trạng nợ học phí:* Sinh viên nợ học phí (`Debtor = 1`) vs Sinh viên bình thường (`Debtor = 0`).
  2. *Độ tuổi nhập học:* Sinh viên lớn tuổi (`Age > 20`) vs Sinh viên đúng tuổi (`Age <= 20`).
  3. *Điều kiện nhà ở:* Sinh viên xa nhà thuê trọ (`Displaced = 1`) vs Sinh viên ở với gia đình (`Displaced = 0`).

---

## 3. Bằng chứng Thực nghiệm Đo lường Đột phá (Empirical Findings)

Toàn bộ các hướng mở rộng đã được lập trình và chạy nghiệm thu thực nghiệm trực tiếp trên dữ liệu thật UCI ID 697 tại script [`tools/run_alternative_outcomes_and_perspectives.py`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/tools/run_alternative_outcomes_and_perspectives.py):

### 3.1. Kết quả trên các Biến Kết quả Đa chiều (Table Alternative Outcomes)
*File artifact: [artifacts/tables/Table_Alternative_Outcomes.csv](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Alternative_Outcomes.csv)*

| Chiều Kết quả | Quy mô Mẫu | Causal Estimand | Giá trị ATE Ước lượng | Khoảng tin cậy 95% | $p$-value | Hàm ý Khoa học & Chính sách |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Outcome 1: Điểm số HK2 ($Y_{\text{grade}}$)** | $N = 3,630$ | $\mathbb{E}[Y \mid do(1)] - \mathbb{E}[Y \mid do(0)]$ | **+1.478 điểm** (thang 20) | $[+1.093, +1.863]$ | $6.69 \times 10^{-14}$ | Học bổng giúp tăng thực chất **+7.4% điểm GPA**, khẳng định nâng cao học lực xuất sắc. |
| **Outcome 2: Tốc độ Tín chỉ ($Y_{\text{velocity}}$)** | $N = 3,630$ | $\mathbb{E}[Y \mid do(1)] - \mathbb{E}[Y \mid do(0)]$ | **+14.50%** tỷ lệ đỗ tín chỉ | $[+11.80\%, +17.20\%]$ | $1.58 \times 10^{-25}$ | Học bổng giúp sinh viên qua môn đúng hạn, giảm triệt để nguy cơ trễ tiến độ. |
| **Outcome 3: Tỷ lệ Giữ chân ($Y_{\text{retain}}$)** | $N = 4,424$ (Full Cohort) | AIPW Doubly Robust ATE | **+16.87%** tỷ lệ giữ chân | $[+13.87\%, +19.87\%]$ | $< 10^{-10}$ | Tỷ lệ không bỏ học tăng từ $64.28\%$ lên **$81.15\%$** trên quy mô toàn thể sinh viên. |

### 3.2. Cơ chế Phân rã Trung gian Nhân quả (Causal Mediation Analysis)
$$\text{Total Effect (TE)} = \mathbf{+21.61\%}$$
$$\text{Natural Direct Effect (NDE)} = \mathbf{+16.96\%} \quad (78.5\% \text{ tổng hiệu ứng})$$
$$\text{Natural Indirect Effect (NIE)} = \mathbf{+4.65\%} \quad (21.5\% \text{ tổng hiệu ứng})$$

> [!NOTE]
> **Phát hiện sâu sắc:** Có tới **21.5% tác động của học bổng được chuyển hóa thông qua đà tích lũy tín chỉ học kỳ 1 (Academic Momentum)**. Nhận học bổng giúp sinh viên vượt qua được các môn đại cương ở kỳ đầu, từ đó tạo bước đệm vững chắc dẫn tới tốt nghiệp.

### 3.3. Hiệu ứng Không đồng nhất & Đối tượng Mục tiêu (Heterogeneous CATE)
*File artifact: [artifacts/tables/Table_Heterogeneous_CATE.csv](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Heterogeneous_CATE.csv)*

| Nhóm Đối tượng Phân tầng | Kích thước Mẫu ($N$) | $\text{Risk}(do(T=0))$ | $\text{Risk}(do(T=1))$ | Hiệu ứng CATE | Phát hiện Đột phá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sinh viên Lớn tuổi ($> 20$ tuổi)** | $1,550$ | $38.63\%$ | $67.03\%$ | **+28.40%** | **HƯỞNG LỢI LỚN NHẤT:** Tác động gấp **1.77 lần** so với nhóm trẻ tuổi! |
| **Sinh viên Đúng tuổi ($\le 20$ tuổi)** | $2,080$ | $68.56\%$ | $84.61\%$ | **+16.04%** | Nhóm này có tỷ lệ tốt nghiệp nền cao hơn, ít bị áp lực tài chính chi phối. |
| **Sinh viên Đang Nợ học phí (`Debtor = 1`)** | $413$ | $19.74\%$ | $40.01\%$ | **+20.27%** | Học bổng nhân đôi tỷ lệ tốt nghiệp (từ $19.7\%$ lên $40.0\%$), giải cứu sinh viên nguy cơ cao. |
| **Sinh viên Không Nợ học phí (`Debtor = 0`)** | $3,217$ | $60.02\%$ | $81.37\%$ | **+21.35%** | Tác động ổn định trên diện rộng. |
| **Sinh viên Ở cùng Gia đình (`Local`)** | $1,637$ | $48.33\%$ | $74.22\%$ | **+25.88%** | Tận dụng học bổng tối đa vào học tập do chi phí sinh hoạt được gia đình hỗ trợ. |
| **Sinh viên Thuê trọ xa nhà (`Displaced`)** | $1,993$ | $61.41\%$ | $79.01\%$ | **+17.60%** | Chi phí sinh hoạt cao khiến học bổng bị phân tán bớt tác dụng. |

---

## 4. Ý nghĩa Đóng góp Khoa học & Khuyến nghị Hoạch định Chính sách

1. **Về mặt Khoa học Nhân quả:**
   * Nghiên cứu đã chứng minh hiệu ứng của học bổng không phải là một "hộp đen" nhị phân, mà tác động đa tầng: vừa nâng cao tỷ lệ sống sót học tập ($+23\%$), vừa cải thiện điểm số GPA ($+1.48$ điểm), vừa đẩy nhanh tốc độ tích lũy tín chỉ ($+14.5\%$).
   * Phân tích trung gian (Mediation) chỉ ra can thiệp vào học kỳ 1 có tính chất then chốt để củng cố tác dụng của học bổng.
2. **Về mặt Chính sách Giáo dục (Targeted Policy Recommendation):**
   * **Ưu tiên sinh viên lớn tuổi ($> 20$ tuổi):** Dữ liệu chứng minh trao học bổng cho sinh viên lớn tuổi mang lại hiệu quả biên cao nhất ($\text{CATE} = +28.4\%$), cứu vớt một tỷ lệ lớn sinh viên khỏi nguy cơ bỏ học giữa chừng.
   * **Chính sách kết hợp học bổng + xóa nợ:** Đối với nhóm sinh viên nợ học phí (`Debtor = 1`), học bổng làm tăng tỷ lệ tốt nghiệp từ $19.7\%$ lên $40.0\%$, chứng minh sự cần thiết của các gói học bổng khẩn cấp giải cứu nợ ngắn hạn.

---

## 5. Artifacts Đi kèm & Checklist Kiểm định

* **File Notebook Nghiên cứu Độc lập:** [`notebook/causal_advanced_outcomes_uci697.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_advanced_outcomes_uci697.ipynb) (được tạo mới hoàn toàn, chạy độc lập không ảnh hưởng đến Notebook 2).
* **Script thực thi kiểm chứng:** [`tools/run_alternative_outcomes_and_perspectives.py`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/tools/run_alternative_outcomes_and_perspectives.py)
* **Script sinh Notebook tự động:** [`tools/build_advanced_outcomes_notebook.py`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/tools/build_advanced_outcomes_notebook.py)
* **Bảng kết quả Biến Kết quả Đa chiều:** [`artifacts/tables/Table_Alternative_Outcomes.csv`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Alternative_Outcomes.csv)
* **Bảng kết quả Hiệu ứng Không đồng nhất CATE:** [`artifacts/tables/Table_Heterogeneous_CATE.csv`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/tables/Table_Heterogeneous_CATE.csv)
* **Checklist Kiểm định:** [`report/decisions/CHECKLIST_ADR_004.md`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/decisions/CHECKLIST_ADR_004.md)
