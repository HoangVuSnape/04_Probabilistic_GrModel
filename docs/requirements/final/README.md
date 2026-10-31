# HƯỚNG DẪN CHI TIẾT YÊU CẦU FINAL (BÁO CÁO KHOA HỌC LATEX & MÃ NGUỒN TÁI LẬP)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện (Nguyen Chi Thien, Ph.D.)  
> **Hình thức đánh giá:** Báo cáo nghiên cứu khoa học toàn văn (10 mục chuẩn mực) + Toàn bộ mã nguồn & artifacts tái lập hoàn chỉnh  
> **Định dạng báo cáo bắt buộc:** **File PDF được biên dịch từ LaTeX** (hoặc file `.docx` học thuật theo Slide 49) kèm mã nguồn `.tex` và file ảnh/bảng biểu  
> **Kiểm duyệt học thuật:** **Kiểm tra đạo văn và tỷ lệ văn bản do AI sinh ra qua hệ thống Turnitin (Turnitin Check AI / Plagiarism)**  
> **Quy tắc điểm nhóm 2 người:** $\text{Score}(\text{First Author}) - \text{Score}(\text{Co-Author}) = 1.0 \text{ điểm}$  
> **Slide tham chiếu trọng tâm:** Slide 45–50 (Rubric Báo cáo 10 điểm), Slide 51 (Checklist tính tái lập), Slide 53–75 (Implementation Guide & Bộ 3 Bảng A–C), Slide 76 (Cấu trúc Notebook & Artifacts), Slide 78 (Minh chứng cho từng phần Báo cáo), Slide 79 (Xử lý sự cố), Slide 80 (Mở rộng tùy chọn).

---

## 1. Mục tiêu và Tiêu chuẩn của Đợt đánh giá Final

Đợt đánh giá Cuối kỳ (Final) yêu cầu nhóm sinh viên hoàn thiện toàn bộ công trình nghiên cứu dưới dạng một **bài báo khoa học hoàn chỉnh chuẩn quốc tế**.

### Các tiêu chuẩn cốt lõi:
1. **Định dạng chuẩn LaTeX / PDF:** Báo cáo được khuyến khích biên dịch từ mã nguồn LaTeX sang file `.pdf` sạch đẹp, công thức toán chuẩn mực, layout chuyên nghiệp (theo Slide 49, chấp nhận file PDF biên dịch từ LaTeX hoặc file `.docx` học thuật).
2. **Kiểm duyệt Turnitin Check AI & Plagiarism:**
   * Hệ thống sẽ quét toàn bộ bài báo trên Turnitin để kiểm tra tỷ lệ trùng lặp văn bản và tỷ lệ văn bản do AI sinh ra (AI Writing Detection).
   * Sinh viên phải tự viết bài báo bằng văn phong học thuật mạch lạc, trích dẫn tài liệu chuẩn xác. Mọi hành vi chép nguyên văn từ ChatGPT/LLM mà không xử lý đều bị trừ điểm nặng hoặc hủy kết quả.
   * **Bắt buộc đính kèm phần Tuyên bố sử dụng AI (AI-Use Disclosure)** minh bạch rõ ràng (Slide 49, 78).
3. **Mã nguồn tái lập hoàn toàn (Full Reproducibility - Slide 51, 76):**
   * Người chấm có thể chạy lại mã nguồn từ đầu trên một môi trường Python/Colab mới tinh (fresh runtime) và thu được các bảng số liệu trùng khớp 100% với báo cáo.
   * Lưu trữ cấu hình tập trung trong một đối tượng duy nhất (`ExperimentConfig`), lưu split indices, checkpoints, ma trận trọng số kề, và kết quả dạng bảng máy đọc được (`.json`, `.csv`).
4. **Hướng tới giải quyết bài toán thực tế trên dữ liệu bảng UCI (Slide 51, 53, 55):**
   * Nghiên cứu phải thực sự giải quyết bài toán can thiệp trên **01 bộ dữ liệu bảng UCI** có tài liệu biến số và trật tự thời gian rõ ràng (ví dụ: Student Performance ID 320 hoặc Student Dropout ID 697).
   * Nếu tập dữ liệu bảng bị thiếu hoặc quá nghèo nàn: Hãy chủ động **thu thập thêm dữ liệu (crawl more data)** hoặc **kỹ nghệ thêm thuộc tính có ý nghĩa (feature engineering)**.

---

## 2. Hệ thống mô hình thực nghiệm (Core Models vs. Optional Extensions)

Theo quy định chuẩn hóa tại **Slide 51, 53 & 75**, hệ thống thực nghiệm phân định rõ:

### 2.1. Ba mô hình cốt lõi bắt buộc (Core Models)
1. **B0 (Baseline CBN):** Học cấu trúc DAG trên tập Train bằng phương pháp truyền thống (`pgmpy` HillClimbSearch / PC / GES hoặc `bnlearn` hc / tabu) kết hợp tri thức miền và ràng buộc thời gian (cấm cạnh ngược thời gian). Khớp phân phối có điều kiện phi nơ-ron bằng phương pháp làm mịn Bayesian smoothing (BDeu prior).
2. **H1 (Neural CPD):** Cố định cấu trúc DAG của B0 cho từng seed. Thay thế một phân phối có điều kiện (ưu tiên biến kết quả $Y$) bằng mạng nơ-ron sâu nhận đầu vào **chỉ gồm các nút cha trực tiếp (parents) của $Y$**, huấn luyện với hàm mất mát Binary Cross-Entropy. Liệt kê các tổ hợp cha rời rạc để tạo bảng CPD chuẩn hóa lắp lại vào mạng Bayes.
3. **H2 (Neural Structure Learning):** Học cấu trúc đồ thị từ gradient liên tục (DAG-GNN / nonlinear NOTEARS) trên cùng biến số và tập Train như B0. Áp dụng ma trận mặt nạ cấm cạnh miền $M$ ($W_{\text{eff}} = M \odot W$) và hàm phạt chu trình $h(W) = 0$. Rerun định danh và khớp CPD non-neural bằng cùng họ ước lượng làm mịn của B0.

### 2.2. Các phần mở rộng tùy chọn (Optional Extensions — Đưa vào Phụ lục / Điểm thưởng)
* **H1+H2 (Biến thể mở rộng kết hợp - Slide 80):** Sử dụng đồ thị đã học được từ H2, sau đó gắn tiếp các phân phối có điều kiện nơ-ron của H1 lên đồ thị này để phân tách đóng góp của cấu trúc đồ thị và cơ chế tham số.
* **Ablations A1–A3 (Nghiên cứu bóc tách kiểm soát - Slide 64–67):**
  * **A1 từ H1:** Loại bỏ Dropout/Regularization (dropout rate = 0).
  * **A2 từ H2:** Gỡ bỏ 1 ràng buộc cấm cạnh thời gian trong ma trận mặt nạ $M$.
  * **A3 từ B0:** Thay Bayesian smoothing (BDeu) bằng Maximum Likelihood Estimation (MLE) thuần túy.
* **Synthetic Tutoring Benchmark & DoWhy GCM (Slide 72–73):** Bài toán phụ đạo học tập 20,000 dòng có chân lý lý thuyết ($ATE=0.30$) và mô phỏng can thiệp GCM.
> *Lưu ý (Slide 44, 51, 53, 75): Synthetic Benchmark và Ablations A1–A3 không phải là yêu cầu bắt buộc của Rubric, không có tiêu chí điểm riêng, có thể trình bày ở Phụ lục (Appendix).*

---

## 3. Thang điểm Rubric 9 phần của Báo cáo Final (10 điểm — Slide 45–50 & 78)

| Phần | Tiêu đề | Điểm | CLO | Nội dung yêu cầu chi tiết (Slide 45–50 & 78) |
| :---: | :--- | :---: | :---: | :--- |
| **2.1** | **Introduction** | **1.0** | CLO1, CLO2 | Nêu câu hỏi nhân quả thực tế trên dữ liệu UCI; lý giải vì sao mô hình mạng Bayes nhân quả (CBN) cơ sở là mốc tham chiếu nền tảng để kiểm thử các thành phần học sâu (DL extensions). |
| **2.2** | **Related Work** | **1.0** | CLO1, CLO2 | Khảo cứu tài liệu về CBN, học tham số nơ-ron (H1), học cấu trúc bằng gradient liên tục (H2, DAG-GNN, NOTEARS), và các thư viện chuẩn mực liên quan. |
| **2.3** | **Quantitative Problem Statement** | **1.0** | CLO1, CLO2, CLO4 | Định nghĩa quần thể mục tiêu, biến $T, Y$, các biến nền $C$; bảng từ điển biến số chi tiết; thời điểm đo lường và can thiệp; mục tiêu can thiệp định lượng $\text{ATE} = \mathbb{E}[Y \mid \text{do}(T=1)] - \mathbb{E}[Y \mid \text{do}(T=0)]$; phân biệt rõ 3 giai đoạn: Discovery $\rightarrow$ Identification $\rightarrow$ Estimation; tuyên bố các giả định nhận diện (unmeasured confounding, consistency, positivity). |
| **2.4** | **Methods and Model Definitions** | **2.0** | CLO3, CLO4 | Đặc tả toán học chặt chẽ cho B0, H1, H2; cấu hình học nơ-ron (kiến trúc, loss, regularizer), cấu hình học đồ thị (hàm mất mát, ma trận mặt nạ $M$, hàm phạt chu trình, ngưỡng threshold); chỉ rõ các khác biệt có kiểm soát giữa các mô hình; công thức can thiệp chuẩn hóa $q_m(t, C_i)$ và $\text{risk}_m(t)$; giao thức ước lượng độ bất định và tính tái lập. |
| **2.5** | **Fair Comparison & Experiment Design** | **1.0** | CLO3, CLO4 | Kiểm toán tính hợp lệ của dữ liệu UCI (eligibility, timing, overlap); khai báo trước tập kiểm thử và tập hiệp biến nền tham chiếu $C_{\text{test}}$; quy tắc chia Train/Val/Test (vd: 70/15/15), fit tiền xử lý chỉ trên Train, đóng băng trên Val trước khi test; quy trình chạy ghép cặp **ít nhất 5 paired runs**; cấu hình 2 phép kiểm định bác bỏ DoWhy (Placebo & Subset refuters). |
| **2.6** | **Prediction and Intervention Results** | **1.0** | CLO3, CLO4 | Trình bày đầy đủ **Bảng A (Prediction), Bảng B (Intervention) và Bảng C (Robustness & Cost)** theo Slide 74–75; định lượng chênh lệch ghép cặp có dấu ($H_1 - B_0$, $H_2 - B_0$), độ bất định (95% CI), kết quả bác bỏ DoWhy, thời gian thực thi và phân tích các trường hợp thất bại. |
| **2.7** | **Conclusions and Limitations** | **1.0** | CLO4, CLO5 | Rút ra kết luận khoa học dựa trên số liệu đo đạc thực tế: Liệu cơ chế nơ-ron hay học đồ thị có thực sự cải thiện dự báo hoặc thay đổi kết luận can thiệp? Thảo luận sâu về biến nhiễu chưa quan sát, tính bao phủ, chi phí tính toán; báo cáo trung thực các kết quả rỗng (null findings) hoặc trường hợp nơ-ron kém hơn baseline. |
| **2.8** | **Abstract** | **1.0** | CLO4, CLO5 | Tóm tắt cô đọng (150–250 từ): Câu hỏi can thiệp UCI, mốc tham chiếu CBN, các mở rộng học sâu, kết quả so sánh chính về dự báo và can thiệp chuẩn hóa, cùng các giới hạn nhân quả cốt lõi. |
| **2.9** | **Questions and Answers (Defense)** | **1.0** | CLO1–CLO5 | Soạn thảo sẵn phần giải trình học thuật bảo vệ các lựa chọn thiết kế: Tính hợp lý của DAG B0, tính định danh Backdoor, tính công bằng trong thực nghiệm, ý nghĩa và giới hạn của các phép kiểm định bác bỏ DoWhy, giải trình bản khai báo sử dụng AI. |

---

## 4. Bộ Ba Bảng Kết quả Chuẩn mực trong Báo cáo (Slide 74–75)

> [!IMPORTANT]
> **Cập nhật quan trọng theo Slide 74–75:**  
> Bộ 3 bảng bắt buộc của Báo cáo Final bao gồm:
> * **Bảng A:** So sánh chỉ số dự báo ngoài mẫu (Prediction Comparison).
> * **Bảng B:** So sánh hiệu ứng can thiệp chuẩn hóa trên tập tham chiếu Test (Interventions & Standardized ATE).
> * **Bảng C:** Đánh giá tính vững bác bỏ DoWhy, độ ổn định và chi phí tính toán (Robustness & Computational Cost).  
> *(Nếu có làm bài toán Synthetic hoặc bóc tách A1–A3, trình bày ở Bảng D/E trong phần Phụ lục).*

### BẢNG A: Đối sánh Năng lực Dự báo Ngoài Mẫu (Standard Prediction Comparison - Slide 74)
*Đánh giá trên cùng tập kiểm thử $Y_{\text{test}}$ qua ít nhất 5 paired runs. Báo cáo từng run và hàng trung bình $\pm$ độ lệch chuẩn:*
| Model | Run ID / Seed | Graph ID | AUROC | Macro-F1 | Brier Score / Log Loss | $\Delta_{\text{Model} - B_0}$ (AUROC) | $\Delta_{\text{Model} - B_0}$ (F1) | Fit Time (s) | Predict Time (s) | Phần cứng |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Run 1..5 | Domain DAG | ... | ... | ... | — | — | ... | ... | CPU/GPU |
| **B0** | **Mean $\pm$ Std** | Domain DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — | — | ... $\pm$ ... | ... $\pm$ ... | — |
| **H1** | Run 1..5 | Fixed B0 DAG | ... | ... | ... | ... | ... | ... | ... | CPU/GPU |
| **H1** | **Mean $\pm$ Std** | Fixed B0 DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — |
| **H2** | Run 1..5 | Learned DAG | ... | ... | ... | ... | ... | ... | ... | CPU/GPU |
| **H2** | **Mean $\pm$ Std** | Learned DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | — |

*Ghi chú:* Đối với bài toán phân loại nhị phân, ngưỡng F1 phải được xác định dựa trên tập Validation. Báo cáo tỷ lệ cân bằng lớp.

---

### BẢNG B: Đánh giá Hiệu ứng Can thiệp Chuẩn hóa trên Tập Test (Interventions & Causal Effects - Slide 75)
*Chuẩn hóa trên cùng tập hiệp biến nền kiểm thử $C_{\text{test}}$ qua ít nhất 5 paired runs:*
| Model | Run ID / Seed | Quần thể tham chiếu & Cỡ mẫu | Công thức định danh & Tập điều chỉnh | $\text{risk}(T=0)$ | $\text{risk}(T=1)$ | $\widehat{\text{ATE}}$ | 95% Confidence Interval | Phương pháp tính CI | Trạng thái Overlap / Đồ thị |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Mean $\pm$ Std | $C_{\text{test}}$ ($n_{\text{test}}$) | Backdoor: $\sum_c P(Y \mid T, c)P(c)$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Bootstrap / Normal | Valid Domain DAG |
| **H1** | Mean $\pm$ Std | $C_{\text{test}}$ ($n_{\text{test}}$) | Neural Backdoor: $\sum_c P_{\text{neural}}(Y \mid T, c)P(c)$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Bootstrap / Normal | Valid Domain DAG |
| **H2** | Mean $\pm$ Std | $C_{\text{test}}$ ($n_{\text{test}}$) | Identified formula trên Learned DAG | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | [...] | Bootstrap / Normal | Learned DAG / Repair |

*Công thức tính:* Với mỗi cá nhân $i \in \text{TEST}$, tính $q_m(t, C_i) = P_m(Y=1 \mid T=t, C_i)$.  
Lấy trung bình: $\text{risk}_m(t) = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} q_m(t, C_i)$ và $\text{ATE}_m = \text{risk}_m(1) - \text{risk}_m(0)$.  
*(Nhãn thực tế $Y_{\text{test}}$ tuyệt đối không tham gia vào phép tính này).*

---

### BẢNG C: Tính vững Bác bỏ và Chi phí Tính toán (Robustness and Computational Cost - Slide 75)
| Model | Hiệu ứng Placebo Refuter ($T_{\text{permute}}$) | Hiệu ứng Data Subset Refuter (80% subset) | Độ tản mạn hiệu ứng 5 runs ($\text{Max} - \text{Min}$) | $\Delta_{\text{ATE}}(\text{Model} - B_0)$ | Tổng thời gian chạy (Fit + Query) | Cấu hình phần cứng |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | ~ 0.000 ($p > 0.05$) | Ổn định ($\Delta < 5\%$) | ... | — | ... s | ... |
| **H1** | ~ 0.000 ($p > 0.05$) | Ổn định ($\Delta < 5\%$) | ... | ... $\pm$ ... | ... s | ... |
| **H2** | ... | ... | ... | ... $\pm$ ... | ... s | ... |

---

## 5. Hướng dẫn Xử lý & Báo cáo Sự cố Thường gặp (Slide 79)

Trong quá trình thực nghiệm và soạn thảo báo cáo, nhóm bắt buộc phải tuân thủ hướng dẫn tại **Slide 79** đối với các sự cố khoa học sau:
1. **Thiếu tính bao phủ hoặc ô giá trị thưa (No overlap / sparse cells):**  
   Báo cáo cụ thể các nhóm bị ảnh hưởng, định lượng mức độ thiếu bao phủ; chỉ được phép thu hẹp tập mục tiêu can thiệp khi có lập luận khoa học chặt chẽ.
2. **Đồ thị H2 không định danh được hoặc vi phạm trật tự thời gian:**  
   Báo cáo rõ ràng tình trạng không định danh hoặc xung đột thời gian; tuyệt đối không báo cáo giá trị số học như thể đó là một hiệu ứng nhân quả đã được kiểm chứng.
3. **Đồ thị H2 xuất hiện chu trình sau khi thresholding:**  
   Áp dụng quy tắc sửa chu trình đã khai báo trước một cách nhất quán (ví dụ: greedy edge removal dựa trên trọng số thấp nhất), hoặc báo cáo lỗi học cấu trúc; lưu trữ cả đồ thị trước và sau khi xử lý chu trình.
4. **Lỗi xác suất bằng 0 trong MLE hoặc tổ hợp cha chưa từng xuất hiện:**  
   Khai báo tường minh quy ước xử lý số học (numerical clipping / epsilon smoothing) tách biệt với kỹ thuật làm mịn xác suất mô hình (Laplace / BDeu).
5. **Mạng nơ-ron quá khớp (overfitting), hiệu ứng mất ổn định hoặc kết quả kém:**  
   Trình bày đường cong mất mát trên tập Train và Validation; không giấu giếm các kết quả bất lợi hoặc kém hơn baseline.
6. **Lỗi tích hợp thư viện kiểm định DoWhy:**  
   Nêu rõ giới hạn kỹ thuật của thư viện, thực hiện kiểm tra đối chứng hỗ trợ và dán nhãn quy trình kiểm tra thay thế một cách minh bạch.

---

## 6. Danh mục Sản phẩm Nộp Cuối kỳ (Final Submission Checklist - Slide 76)

- [ ] **Báo cáo toàn văn:** File PDF bài báo cáo hoàn chỉnh được biên dịch từ LaTeX (hoặc file `.docx` học thuật chuẩn mực theo Slide 49).
- [ ] **Mã nguồn văn bản:** Toàn bộ thư mục mã nguồn LaTeX (`.tex`, `.bib`, thư mục hình ảnh `figures/`).
- [ ] **Bản tuyên bố AI (AI-Use Disclosure):** Khai báo minh bạch việc sử dụng công cụ AI (ChatGPT, Gemini, Copilot...) trong bài báo.
- [ ] **Kiểm tra Turnitin:** Đảm bảo tỷ lệ trùng lặp và tỷ lệ AI nằm trong ngưỡng cho phép trước khi nộp chính thức.
- [ ] **Mã nguồn tái lập:** Notebook Python (`.ipynb`) sạch sẽ, chạy từ đầu đến cuối không lỗi trên môi trường Colab mới tinh (fresh runtime restart), lưu giữ đầy đủ outputs.
- [ ] **File môi trường:** File khai báo thư viện (`requirements.txt` hoặc `environment.yml`) với phiên bản phần mềm chính xác.
- [ ] **Thư mục Artifacts:** Lưu trữ đầy đủ split indices, checkpoints nơ-ron, ma trận trọng số kề H2, và các file kết quả dạng bảng máy đọc được (`.json`, `.csv`).
- [ ] **Tính nhất quán:** Đảm bảo số liệu, tên biến, đồ thị và kết quả trong Báo cáo, Notebook và Slide thuyết trình hoàn toàn trùng khớp với nhau.
