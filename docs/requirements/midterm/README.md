# HƯỚNG DẪN CHI TIẾT YÊU CẦU MIDTERM (THUYẾT TRÌNH & DEMO NOTEBOOK)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện (Nguyen Chi Thien, Ph.D.)  
> **Thời gian làm bài:** **2 tuần**  
> **Hình thức đánh giá:** Thuyết trình slide + Trình diễn trực tiếp mã nguồn trên Notebook (Google Colab / Jupyter) qua hình thức **Trực tuyến (Online)**  
> **Cơ chế duyệt đề tài:** **Accept / Reject** (Chấp thuận hoặc Từ chối đề tài dựa trên tính hợp lý của câu hỏi nhân quả và cấu trúc đồ thị)  
> **Quy mô nhóm:** 1 – 2 học viên (Group of 1–2 students)  
> **Quy tắc tính điểm nhóm:** $\text{Score}(\text{First Author}) - \text{Score}(\text{Co-Author}) = 1.0 \text{ điểm}$  
> **Yêu cầu nộp bài:** **Tất cả các thành viên trong nhóm đều phải nộp file `.ipynb` của nhóm lên hệ thống.**  
> **Tài liệu tham khảo gốc:** [Google Drive Slide Hướng dẫn (80 slides)](https://drive.google.com/file/d/11vuIrPn783XorbLwZcXEGU3GWCNUEVdx/view?usp=sharing) | [Gemini NotebookLM](https://notebook.google.com/notebook/33dd2d2a-8013-4591-a1c2-bc6c0d0fd32a)  
> **Slide tham chiếu trọng tâm:** Slide 39–44 (Phạm vi & Rubric Midterm), Slide 51, 53–75 (Implementation Guide, Tables A–C), Slide 76–77 (Notebook & Kịch bản Demo 6 bước), Slide 79 (Xử lý lỗi).

---

## 1. Mục tiêu và Ý nghĩa của Đợt đánh giá Midterm

Đợt đánh giá Midterm kéo dài trong **2 tuần làm việc**, nhằm mục đích:
1. **Kiểm tra tính hợp lệ và phê duyệt đề tài (Accept / Reject):** Giảng viên thẩm định bài toán nhân quả, dữ liệu lựa chọn từ UCI, tính hợp lý của biến can thiệp $T$, biến kết quả $Y$, các biến đồng biến nền $Z$ đo lường trước $T$, và trật tự thời gian của các cạnh trong DAG.
2. **Nền tảng lý thuyết vững chắc (Foundations — 4 điểm):** Hiểu rõ bản chất biểu diễn nhân quả, phân rã xác suất, phân biệt rạch ròi giữa quan sát $P(Y \mid T=t)$ và can thiệp $P(Y \mid \text{do}(T=t))$, cùng giới hạn của mạng nơ-ron đối với định danh nhân quả.
3. **Năng lực triển khai thực nghiệm chuẩn mực (Application — 6 điểm):** Cài đặt thành công 3 mô hình B0, H1, H2 trên 01 bộ dữ liệu bảng UCI; chạy ghép cặp trên ít nhất 5 lần chia dữ liệu (paired runs $\ge 5$); chuẩn hóa hiệu ứng trên cùng một tập tham chiếu kiểm thử $C_{\text{test}}$; tích hợp kiểm định bác bỏ DoWhy.
4. **Trình diễn tương tác trực tiếp theo kịch bản chuẩn (Live Colab Demo — Slide 77):** Trình diễn tuần tự 6 bước thực tế trên Colab, minh chứng đối soát từ tham số cấu hình, checkpoint, hàm tính toán đến bảng kết quả.

---

## 2. Các nguyên tắc sống còn Giảng viên lưu ý trong Midterm (Cập nhật 80 slides)

1. **"Đo can thiệp, không đo quan sát":**
   * Giảng viên đặc biệt nhấn mạnh: Tuyệt đối không đưa ra khuyến nghị chính sách dựa trên xác suất điều kiện quan sát $P(Y \mid T=t)$.
   * Phải sử dụng công thức can thiệp đã qua phẫu thuật đồ thị (graph surgery / do-calculus): $P(Y \mid \text{do}(T=t))$.
2. **"Chuẩn hóa quần thể tham chiếu trên tập Test (Common Test Reference Cohort)":**
   * Điểm số dự báo (AUROC, F1, Brier) được chấm dựa trên nhãn thực tế $Y_{\text{test}}$.
   * **Tuyệt đối không dùng nhãn $Y_{\text{test}}$ để tính hiệu ứng can thiệp!** Hiệu ứng can thiệp của mọi mô hình ($m \in \{B_0, H_1, H_2\}$) phải được tính bằng cách lấy trung bình nguy cơ điều kiện $q_m(t, C_i) = P_m(Y=1 \mid T=t, C_i)$ trên cùng một tập hiệp biến nền tiền can thiệp $C_{\text{test}}$:
     $$\text{risk}_m(t) = \frac{1}{n_{\text{test}}} \sum_{i \in \text{TEST}} q_m(t, C_i), \quad \text{ATE}_m = \text{risk}_m(1) - \text{risk}_m(0)$$
3. **"Tách bạch đánh giá Dự báo và Đánh giá Can thiệp":**
   * Mạng nơ-ron khớp tốt dữ liệu quan sát (loss thấp, AUROC cao) **không chứng minh** được nó ước lượng đúng hiệu ứng nhân quả.
   * Khoảng tin cậy hẹp không đồng nghĩa với tính chân lý nhân quả. Cần bảo vệ bằng các giả định nhận diện hình thức và bài kiểm thử bác bỏ DoWhy.
4. **"Chạy ghép cặp ít nhất 5 lần — Chạy 1 lần là không đủ (Slide 68)":**
   * Phải thực hiện trên **ít nhất 5 lần chạy ghép cặp (paired runs $\ge 5$)** dùng chung chỉ số Train/Val/Test để báo cáo giá trị trung bình $\pm$ độ lệch chuẩn, khoảng tin cậy 95% và chênh lệch ghép cặp có dấu ($H_1 - B_0$, $H_2 - B_0$).
5. **"Học đồ thị gradient có ma trận mặt nạ cấm cạnh (Gradient-based DAG with Domain Mask - Slide 63)":**
   * Trong H2: Dùng thuật toán tối ưu vi phân liên tục (DAG-GNN / NOTEARS), dùng gradient học ma trận trọng số $W$, áp dụng ma trận mặt nạ cạnh cấm $M$ ($W_{\text{eff}} = M \odot W$) để triệt tiêu các cạnh vi phạm thời gian, duy trì ràng buộc phi chu trình $h(W) = 0$.
   * **ChatGPT là KHÔNG ĐỦ:** Hỏi ChatGPT gợi ý đồ thị không được coi là thuật toán học cấu trúc.
6. **"Hướng tới giải bài toán thực tế trên 01 bộ dữ liệu bảng UCI":**
   * Mục tiêu là trả lời câu hỏi thực tiễn trên dữ liệu bảng.
   * **Synthetic Tutoring Benchmark và các biến thể bóc tách A1–A3 là TÙY CHỌN (Optional Extensions - Slide 44, 51, 53)**, không nằm trong các tiêu chí chấm điểm bắt buộc của Rubric.

---

## 3. Thang điểm Rubric chi tiết Midterm (10 điểm — Chuẩn hóa theo Slide 42 & 43)

### PHẦN I: NỀN TẢNG LÝ THUYẾT (FOUNDATIONS — 4 ĐIỂM)

* **Tiêu chí 1: Biểu diễn nhân quả (Causal Representation) — 1.0 điểm [CLO1, CLO2]** (Slide 42, 54)
  * Viết câu hỏi can thiệp nhân quả định lượng (quần thể, treatment $T$, outcome $Y$, thời điểm can thiệp).
  * Lập bảng từ điển biến số (tên biến, kiểu mã hóa, thời điểm đo lường, vai trò nhân quả, tỷ lệ missing).
  * Vẽ đồ thị DAG cho B0, bảo vệ tính hợp lý của chiều mũi tên theo trật tự thời gian từ tài liệu khoa học.
  * Viết công thức phân rã xác suất đồng thời (Factorization).
  * Tuyên bố rõ ràng 3 giả định nhận diện: No unmeasured confounding, consistency, positivity/overlap.

* **Tiêu chí 2: Vai trò của Học tham số & Học sâu (Learning and DL Roles) — 2.0 điểm [CLO1, CLO2]** (Slide 42, 57, 60, 62)
  * Trình bày phương pháp làm mịn tham số Bayesian smoothing (BDeu prior) cho B0.
  * Trình bày kiến trúc nơ-ron thay thế CPD trong H1 (đầu vào chỉ gồm parents của $Y$, hàm mất mát BCE, tạo bảng CPD chuẩn hóa).
  * Trình bày cơ chế học cấu trúc đồ thị bằng gradient liên tục và ma trận mặt nạ cấm cạnh trong H2.
  * Phân tích sâu sắc: Khả năng khớp dữ liệu quan sát của Deep Learning không thay thế được bước định danh nhân quả hình thức.

* **Tiêu chí 3: Suy diễn quan sát và can thiệp (Inference) — 1.0 điểm [CLO1, CLO2]** (Slide 42, 59, 61)
  * Phân biệt rạch ròi bằng công thức giữa xác suất quan sát $P(Y \mid T=t)$ và xác suất can thiệp $P(Y \mid \text{do}(T=t))$.
  * Thể hiện cơ chế phân rã cắt cụt (Truncated Factorization): Xóa bỏ các cạnh hướng vào $T$ khi can thiệp.
  * Áp dụng tiêu chuẩn Backdoor để tìm tập điều chỉnh hợp lệ và biểu diễn công thức tính ATE chuẩn hóa trên quần thể tham chiếu $C_{\text{test}}$.

---

### PHẦN II: ỨNG DỤNG VÀ THỰC NGHIỆM (APPLICATION — 6 ĐIỂM)

* **Tiêu chí 4: Lựa chọn bài toán & Dữ liệu UCI (Problem and UCI data) — 1.0 điểm [CLO3]** (Slide 43, 55)
  * Sử dụng 01 bộ dữ liệu bảng từ UCI (ví dụ: Student Performance ID 320 hoặc Student Dropout ID 697).
  * Lưu trữ nguồn gốc: URL tải, Dataset ID / DOI, ngày tải, bản quyền, checksum file thô.
  * Báo cáo lưu đồ lọc dữ liệu (eligibility flow): số dòng trước và sau lọc, kiểm tra trùng lặp, biến mục tiêu, rò rỉ dữ liệu.
  * Kiểm toán trật tự thời gian (timing audit) và kiểm tra độ bao phủ can thiệp (treatment overlap), chỉ ra các nhóm thưa dữ liệu.

* **Tiêu chí 5: Cài đặt và tích hợp mã nguồn (Implementation) — 1.0 điểm [CLO3]** (Slide 43, 76)
  * Mã nguồn Python sạch, chạy thông suốt 100% trên Colab từ đầu đến cuối không lỗi (fresh restart run).
  * Tối thiểu **3 mô hình B0, H1, H2** hoạt động trơn tru (khuyến khích mở rộng tùy chọn H1+H2).
  * Sử dụng một đối tượng cấu hình duy nhất (`ExperimentConfig`), lưu giữ đầy đủ checkpoints, ma trận trọng số kề và artifacts máy đọc được.

* **Tiêu chí 6: Đối sánh dự báo ngoài mẫu (Prediction comparison) — 1.0 điểm [CLO3, CLO4]** (Slide 43, 74)
  * Dùng chung phân chia dữ liệu Train/Val/Test (vd: 70/15/15) và thực hiện trên **ít nhất 5 paired runs**.
  * Tiền xử lý chỉ fit trên tập Train; tập Validation dùng để dừng sớm / chọn ngưỡng đồ thị; tập Test đóng băng đánh giá.
  * Trình bày **Bảng A (Slide 74)**: Báo cáo AUROC, Macro-F1, Brier Score (hoặc Log Loss) đối với nhị phân; MAE/RMSE đối với liên tục; kèm chênh lệch ghép cặp có dấu ($H_1 - B_0$, $H_2 - B_0$).

* **Tiêu chí 7: Đối sánh hiệu ứng can thiệp (Intervention comparison) — 1.0 điểm [CLO3, CLO4]** (Slide 43, 61, 75)
  * Dùng chung tập hiệp biến nền kiểm thử $C_{\text{test}}$ làm quần thể tham chiếu để chuẩn hóa nguy cơ cho cả B0, H1, H2.
  * Trình bày **Bảng B (Slide 75)**: Báo cáo nguy cơ can thiệp $\text{risk}_m(0)$, $\text{risk}_m(1)$, hiệu ứng trung bình $\text{ATE}_m$ và khoảng tin cậy 95% kèm phương pháp tính (bootstrap / normal / delta method).
  * Tuyên bố rõ tập điều chỉnh và trạng thái định danh của đồ thị tương ứng.

* **Tiêu chí 8: Tính vững và Chi phí tính toán (Robustness and cost) — 1.0 điểm [CLO3, CLO4]** (Slide 43, 70, 75)
  * Thực thi **ít nhất 2 bài kiểm tra bác bỏ DoWhy**:
    1. `placebo_treatment_refuter`: Hoán vị ngẫu nhiên nhãn $T$ (hiệu ứng kỳ vọng triệt tiêu về 0).
    2. `data_subset_refuter`: Lấy mẫu lại 80% tập dữ liệu để kiểm tra độ nhạy của hiệu ứng.
  * Trình bày **Bảng C (Slide 75)**: Báo cáo kết quả kiểm định bác bỏ, độ biến thiên hiệu ứng qua các lần chạy ghép cặp (paired effect spread), thời gian huấn luyện/suy diễn (fit/query runtime) và cấu hình phần cứng.

* **Tiêu chí 9: Kết luận & Trả lời phản biện (Conclusions and Q&A) — 1.0 điểm [CLO4]** (Slide 43, 48, 77, 79)
  * Rút ra kết luận khách quan dựa trên số liệu thực nghiệm đo đạc: Không thiên vị cho rằng Deep Learning luôn tốt hơn.
  * Phân tích trung thực các trường hợp thất bại (ô xác suất 0, đồ thị H2 chu trình hoặc vi phạm trật tự thời gian, cảnh báo rỗng - Slide 79).
  * Bảo vệ tự tin câu hỏi nhân quả, cấu trúc DAG và phương pháp thực nghiệm trước Giảng viên.

> [!NOTE]
> **Về bài toán Synthetic Tutoring Benchmark & Nghiên cứu bóc tách A1–A3:**
> Theo Slide 44, 51, 53 & 75: Thí nghiệm dữ liệu nhân tạo và bóc tách A1–A3 là **phần mở rộng tùy chọn (Optional Extensions)**, không bắt buộc cho điểm số tối đa của Rubric Midterm. Nếu nhóm có triển khai, có thể trình bày ngắn gọn ở phần phụ lục để tăng tính thuyết phục khoa học.

---

## 4. Kịch bản Trình diễn Colab Demo Online (6 bước bắt buộc theo Slide 77)

Khi bảo vệ trực tuyến trên lớp, nhóm phải chia sẻ màn hình Google Colab và thực hiện đúng **trình tự 6 bước chuẩn mực (Slide 77)**:

```mermaid
sequenceDiagram
    autonumber
    actor Team as Nhóm sinh viên
    actor Lecturer as Giảng viên (Online)
    participant Colab as Màn hình Colab Notebook

    Team->>Lecturer: 1. Nêu câu hỏi nhân quả UCI, trật tự thời gian, đồ thị & giả định; chỉ rõ tập tham chiếu C_test và giá trị T (control/treatment)
    Team->>Colab: 2. Giải thích phân định dữ liệu: TRAIN (huấn luyện), VAL (chọn ngưỡng/dừng sớm), TEST (đánh giá dự báo & chuẩn hóa can thiệp)
    Team->>Colab: 3. Chỉ rõ vị trí thay thế CPD nơ-ron trong H1; mở ma trận trọng số kề W, mặt nạ mask M và đồ thị học được của H2 kèm kết quả DoWhy identify
    Team->>Colab: 4. Chọn một số profile cá nhân C_i từ tập Test, hiển thị q_m(0, C_i) và q_m(1, C_i); chạy lệnh lấy trung bình trên toàn bộ C_test để ra risk(t) và ATE
    Team->>Colab: 5. Trình chiếu Bảng A (dự báo TEST), Bảng B (can thiệp & 95% CI), Bảng C (độ ổn định, DoWhy Placebo/Subset refuters & runtime)
    Team->>Lecturer: 6. Phân tích 1 ca thất bại / kết quả rỗng / giới hạn nhân quả; truy vết trực tiếp từ số liệu hiển thị về tham số config và dòng code tương ứng
```

### Chi tiết hành động tại từng bước:
1. **Bước 1 (Causal Question & Context):** Nêu ngắn gọn 1 câu về câu hỏi can thiệp, biến $T, Y$, các biến nền $C$, tập tham chiếu $C_{\text{test}}$, giá trị đối chứng $T=0$ và can thiệp $T=1$.
2. **Bước 2 (Data Role Transparency):** Minh chứng rõ ràng: Mọi phép chuẩn hóa/mã hóa chỉ `fit` trên tập Train; tập Val dùng để chọn ngưỡng/stopping; tập Test giữ nguyên vẹn.
3. **Bước 3 (Architectures H1 & H2):**
   * Cho xem kiến trúc MLP của H1 (đầu vào chỉ là parents của $Y$).
   * Cho xem ma trận mặt nạ $M$ của H2 (các ô cấm vi phạm thời gian có giá trị 0) và đồ thị DAG học được sau khi thresholding.
4. **Bước 4 (Intervention Calculation & Live Query):**
   * Hiển thị bảng tính nguy cơ điều kiện $q_m(t, C_i)$ cho một vài cá nhân mẫu trên Test.
   * Chạy lệnh tính trung bình trên tập tham chiếu $C_{\text{test}}$: $\text{risk}_m(t) = \frac{1}{n_{\text{test}}} \sum_{i} q_m(t, C_i)$ để ra rủi ro và $\text{ATE}_m$.
   * Giải thích trực quan tại sao can thiệp triệt tiêu các mũi tên hướng vào $T$.
5. **Bước 5 (Results Comparison Tables A–C):**
   * Chiếu Bảng A: So sánh AUROC, Macro-F1, Brier, chênh lệch $H_1 - B_0$, $H_2 - B_0$.
   * Chiếu Bảng B: So sánh rủi ro can thiệp và ATE kèm khoảng tin cậy 95%.
   * Chiếu Bảng C: Kết quả 2 bài kiểm định bác bỏ DoWhy (Placebo & Subset refuters) và thời gian thực thi.
6. **Bước 6 (Failure Analysis & Code Traceability):**
   * Trình bày trung thực một giới hạn, kết quả rỗng (null finding), hoặc trường hợp thất bại (ví dụ: nhóm thưa không có mẫu $T=1$, hoặc đồ thị H2 học thừa cạnh).
   * Truy vết ngay lập tức từ kết quả trên bảng về dòng code và file artifact lưu trữ tương ứng.

---

## 5. Quy cách Bộ 3 Bảng Kết quả Bắt buộc (Slide 74–75)

### BẢNG A: Đối sánh Năng lực Dự báo Ngoài Mẫu (Standard Prediction Comparison - Slide 74)
*Đánh giá trên tập kiểm thử chung $Y_{\text{test}}$ qua ít nhất 5 paired runs:*
| Model | Run ID / Seed | Graph ID | AUROC | Macro-F1 | Brier Score / Log Loss | $\Delta_{\text{Model} - B_0}$ (AUROC) | Fit Time (s) | Predict Time (s) | Hardware |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Run 1..5 | Domain DAG | ... | ... | ... | — | ... | ... | CPU/GPU |
| **B0** | **Mean $\pm$ Std** | Domain DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — | ... $\pm$ ... | ... $\pm$ ... | — |
| **H1** | Run 1..5 | Fixed B0 DAG | ... | ... | ... | ... | ... | ... | CPU/GPU |
| **H1** | **Mean $\pm$ Std** | Fixed B0 DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — |
| **H2** | Run 1..5 | Learned DAG | ... | ... | ... | ... | ... | ... | CPU/GPU |
| **H2** | **Mean $\pm$ Std** | Learned DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — |

---

### BẢNG B: Đánh giá Hiệu ứng Can thiệp Chuẩn hóa trên Tập Test (Interventions & Causal Effects - Slide 75)
*Chuẩn hóa trên cùng tập hiệp biến nền $C_{\text{test}}$ qua ít nhất 5 paired runs:*
| Model | Run ID / Seed | Cohort ID & $n_{\text{test}}$ | Công thức định danh & Tập điều chỉnh | $\text{risk}(T=0)$ | $\text{risk}(T=1)$ | $\widehat{\text{ATE}}$ | 95% Confidence Interval | Phương pháp CI | Trạng thái Overlap / Graph |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Mean $\pm$ Std | Test Cohort ($N$) | Backdoor $\{Z_1..Z_k\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Normal / Bootstrap | Valid DAG |
| **H1** | Mean $\pm$ Std | Test Cohort ($N$) | Backdoor $\{Z_1..Z_k\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Normal / Bootstrap | Valid DAG |
| **H2** | Mean $\pm$ Std | Test Cohort ($N$) | Identified Formula / Adj Set | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Normal / Bootstrap | Learned DAG / Repair |

---

### BẢNG C: Tính vững Bác bỏ và Chi phí Tính toán (Robustness and Computational Cost - Slide 75)
| Model | Placebo Refuter Effect ($T_{\text{permute}}$) | Data Subset Refuter Effect (80% subset) | Paired Effect Spread ($\text{Max} - \text{Min}$) | $\Delta_{\text{ATE}}(\text{Model} - B_0)$ | Tổng thời gian chạy (Fit + Query) | Cấu hình phần cứng (RAM/GPU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | ~ 0.000 ($p > 0.05$) | Ổn định ($\Delta < 5\%$) | ... | — | ... s | ... |
| **H1** | ~ 0.000 ($p > 0.05$) | Ổn định ($\Delta < 5\%$) | ... | ... $\pm$ ... | ... s | ... |
| **H2** | ... | ... | ... | ... $\pm$ ... | ... s | ... |

---

## 6. Hướng dẫn Xử lý 6 Sự cố Thường gặp (Slide 79)

Khi chạy mã nguồn và làm slide báo cáo, nếu gặp các sự cố dưới đây, nhóm phải tuân thủ đúng quy định tại **Slide 79**:
1. **Thiếu tính bao phủ hoặc có nhóm giá trị rỗng (No overlap / sparse cells):** Báo cáo số lượng và tỷ lệ nhóm bị ảnh hưởng, định lượng mức độ nghiêm trọng; chỉ được thu hẹp tập mục tiêu can thiệp khi có giải trình khoa học thỏa đáng.
2. **Đồ thị H2 không thể định danh hoặc vi phạm trật tự thời gian:** Báo cáo trung thực tình trạng này; tuyệt đối không ngụy tạo kết quả số học thành hiệu ứng nhân quả đã kiểm chứng.
3. **Đồ thị H2 xuất hiện chu trình sau khi thresholding:** Áp dụng quy tắc cắt chu trình đã khai báo nhất quán từ trước, hoặc ghi nhận lỗi học cấu trúc; lưu trữ cả đồ thị thô và đồ thị sau sửa chữa.
4. **Lỗi xác suất bằng 0 trong MLE hoặc tổ hợp cha chưa từng xuất hiện:** Khai báo quy ước xử lý số học tách biệt với kỹ thuật làm mịn xác suất.
5. **Mạng nơ-ron quá khớp (overfitting), hiệu ứng can thiệp mất ổn định:** Vẽ đường cong training/validation loss qua các epoch; không giấu giếm các kết quả bất lợi.
6. **Lỗi không tương thích thư viện kiểm định DoWhy:** Báo cáo giới hạn kỹ thuật thực tế, thực hiện bài kiểm tra đối chứng hỗ trợ và dán nhãn quy trình thay thế rõ ràng.
