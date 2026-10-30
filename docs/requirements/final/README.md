# HƯỚNG DẪN CHI TIẾT YÊU CẦU FINAL (BÁO CÁO KHOA HỌC LATEX & MÃ NGUỒN TÁI LẬP)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện (Nguyen Chi Thien, Ph.D.)  
> **Hình thức đánh giá:** Báo cáo nghiên cứu khoa học toàn văn + Bộ mã nguồn tái lập hoàn chỉnh  
> **Định dạng báo cáo bắt buộc:** **File PDF được biên dịch từ LaTeX** (kèm toàn bộ mã nguồn `.tex` và file ảnh/bảng biểu)  
> **Kiểm duyệt học thuật:** **Kiểm tra đạo văn và tỷ lệ AI qua hệ thống Turnitin (Turnitin Check AI / Plagiarism)**  
> **Quy tắc điểm nhóm 2 người:** $\text{Score}(\text{First Author}) - \text{Score}(\text{Co-Author}) = 1.0 \text{ điểm}$  
> **Slide tham chiếu:** Slide 24, 30–35, 36, 56–58, 60, 61, 62  

---

## 1. Mục tiêu và Tiêu chuẩn của Đợt đánh giá Final

Đợt đánh giá Cuối kỳ (Final) yêu cầu nhóm sinh viên hoàn thiện toàn bộ công trình nghiên cứu dưới dạng một **bài báo khoa học hoàn chỉnh chuẩn quốc tế**.

### Các tiêu chuẩn cốt lõi:
1. **Định dạng chuẩn LaTeX:** Báo cáo phải được soạn thảo bằng LaTeX và biên dịch ra file `.pdf` sạch đẹp, công thức toán chuẩn xác.
2. **Kiểm duyệt Turnitin Check AI:**
   * Hệ thống sẽ quét toàn bộ báo cáo trên Turnitin để kiểm tra tỷ lệ trùng lặp và tỷ lệ văn bản do AI sinh ra (AI Writing Detection).
   * Sinh viên phải tự viết bài báo, sử dụng văn phong học thuật chuẩn mực, trích dẫn tài liệu đầy đủ. Mọi hành vi sao chép nguyên văn từ ChatGPT/LLM mà không xử lý sẽ bị trừ điểm nặng hoặc hủy kết quả.
   * Kèm theo phần **Tuyên bố sử dụng AI (AI-use Statement)** minh bạch rõ ràng.
3. **Mã nguồn tái lập hoàn toàn (Full Reproducibility):**
   * Người chấm có thể chạy lại mã nguồn từ đầu trên một môi trường Python/Colab mới tinh và thu được các bảng số liệu trùng khớp 100% với báo cáo.
4. **Hướng tới giải quyết bài toán thực tế (Problem-Driven, Not Tool-Driven):**
   * Nghiên cứu phải thực sự giải bài toán thực tế. Nếu tập dữ liệu ban đầu thiếu biến số hoặc quá nghèo nàn, nhóm được khuyến khích **thu thập thêm dữ liệu (crawl more data)** hoặc **bổ sung thêm các thuộc tính có ý nghĩa (add features to have more meaning)**.

---

## 2. Hệ thống mô hình thực nghiệm (Tối thiểu 3 mô hình + Mở rộng)

Nhóm bắt buộc phải cài đặt **tối thiểu 3 mô hình B0, H1, H2** và **3 biến thể bóc tách A1, A2, A3**, đồng thời được khuyến khích mở rộng mô hình lai **H1+H2**:

1. **B0 (Baseline CBN):** Domain DAG + CPD phi nơ-ron (Bayesian smoothing với BDeu prior).
2. **H1 (Neural CPD):** Cố định DAG của B0, dùng mạng nơ-ron sâu học phân phối $P(Y \mid \text{Parents}(Y))$.
3. **H2 (Neural Structure Learning):** Học cấu trúc đồ thị từ gradient liên tục (DAG-GNN / Neural NOTEARS) kèm **ma trận mặt nạ cạnh cấm (domain mask)**. Dùng DoWhy tái định danh và khớp CPD non-neural.
4. **H1+H2 (Biến thể mở rộng kết hợp — Khuyến khích):** Sử dụng đồ thị đã học được từ H2, sau đó gắn tiếp các phân phối có điều kiện nơ-ron của H1 lên đồ thị này để kết hợp sức mạnh của cả hai.
5. **Ablations A1–A3:** Bóc tách kiểm soát ("One change only"):
   * **A1 từ H1:** Bỏ dropout / regularization.
   * **A2 từ H2:** Gỡ bỏ 1 ràng buộc cấm cạnh thời gian trong ma trận mask.
   * **A3 từ B0:** Thay Bayesian smoothing bằng Maximum Likelihood Estimation thuần túy.

---

## 3. Thang điểm Rubric 9 phần của Báo cáo (10 điểm)

Theo quy định tại **Slide 30–35 & 60**:

| Phần | Nội dung yêu cầu | Điểm | CLO |
| :---: | :--- | :---: | :---: |
| **2.1** | **Introduction (Mở đầu):** Nêu câu hỏi nhân quả thực tế; lý giải vì sao CBN là mốc tham chiếu nền tảng để kiểm thử học sâu. | **1.0** | CLO1, CLO2 |
| **2.2** | **Related Work (Tổng quan nghiên cứu):** Khảo cứu CBN, học tham số nơ-ron (H1), học cấu trúc bằng gradient (H2, DAG-GNN, NOTEARS). | **1.0** | CLO1, CLO2 |
| **2.3** | **Quantitative Problem Statement (Phát biểu toán học):** Quần thể, biến $T, Y, Z$, bảng từ điển biến số, công thức ATE, phân biệt 3 giai đoạn (Discovery $\rightarrow$ Identification $\rightarrow$ Estimation), các giả định Backdoor. | **1.0** | CLO1, CLO2, CLO4 |
| **2.4** | **Proposed Methods & Ablations (Phương pháp & Bóc tách):** Đặc tả toán học B0, H1, H2 (gradient và ma trận mặt nạ $M$), H1+H2; định nghĩa chặt chẽ A1, A2, A3; mã giả thuật toán và cơ chế xử lý lỗi. | **2.0** | CLO3, CLO4 |
| **2.5** | **Experimental Design (Thiết kế thực nghiệm):** Kiểm toán dữ liệu UCI (hoặc dữ liệu bổ sung), tỷ lệ chia Train/Val/Test, quy trình chạy lặp **ít nhất 5 seeds**, bài toán Synthetic Benchmark (ATE=0.30), cấu hình 2 phép kiểm định DoWhy. | **1.0** | CLO3, CLO4 |
| **2.6** | **Experimental Results (Kết quả thực nghiệm):** Trình bày đầy đủ **Bảng A** (UCI), **Bảng B** (Synthetic truth), **Bảng C** (Ablations với $\Delta$), chỉ số dự báo ngoài mẫu, thời gian chạy và kết quả kiểm thử bác bỏ DoWhy. | **1.0** | CLO3, CLO4 |
| **2.7** | **Conclusions & Limitations (Kết luận & Hạn chế):** Đúc kết thành phần nào mang lại hiệu quả thực chất, khi nào B0 đã đủ tốt; thảo luận sâu về biến nhiễu ẩn, tính bao phủ dữ liệu và chi phí tính toán. Báo cáo trung thực kết quả rỗng. | **1.0** | CLO4, CLO5 |
| **2.8** | **Abstract (Tóm tắt bài báo):** Cô đọng từ 150–250 từ: bài toán, phương pháp B0/H1/H2, kết quả ATE chính, phát hiện bóc tách và giới hạn nhân quả. | **1.0** | CLO4, CLO5 |
| **2.9** | **Questions & Answers (Giải trình & Bảo vệ):** Trả lời chất vấn về tính hợp lý của DAG, nhận diện Backdoor, sự công bằng trong thực nghiệm, và ý nghĩa của các bài kiểm tra bác bỏ DoWhy. | **1.0** | CLO1–CLO5 |

---

## 4. Ba bảng kết quả bắt buộc trong Báo cáo

### BẢNG A: Đối sánh Mô hình Cơ sở và Mô hình Lai (Trên dữ liệu thực tế UCI)
*Chạy trên ít nhất 5 hạt giống ngẫu nhiên (seeds $\ge 5$), báo cáo giá trị trung bình $\pm$ độ lệch chuẩn:*
| Mô hình | Trạng thái định danh | Tập điều chỉnh | $P(Y=1 \mid \text{do}(T=0))$ | $P(Y=1 \mid \text{do}(T=1))$ | ATE (95% CI) | AUROC | Macro-F1 | Brier Score | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Identified (Backdoor) | $\{Z_1, Z_2, ...\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... | ... | ... | ... |
| **H1** | Identified (Backdoor) | $\{Z_1, Z_2, ...\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... | ... | ... | ... |
| **H2** | Identified / Failed | $\{...\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... | ... | ... | ... |
| **H1+H2** *(opt)* | Identified / Failed | $\{...\}$ | ... $\pm$ ... | ... $\pm$ ... | ... $\pm$ ... | ... | ... | ... | ... |

### BẢNG B: Kết quả trên Bài toán Mô phỏng có Chân lý (Synthetic Truth Benchmark)
*Chân lý lý thuyết: $\text{ATE}_{\text{true}} = 0.300$, rủi ro can thiệp là $0.300$ và $0.600$:*
| Mô hình | Cỡ mẫu $N$ | $\widehat{\text{ATE}}$ trung bình | True ATE | Sai số tuyệt đối ($|\widehat{\text{ATE}} - 0.30|$) | Độ lệch chuẩn qua 5 seeds | Sai số cạnh đồ thị (Graph Edit Distance) | Thời gian chạy (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | 20,000 | ... | 0.300 | ... | ... | N/A (Fixed DAG) | ... |
| **H1** | 20,000 | ... | 0.300 | ... | ... | N/A (Fixed DAG) | ... |
| **H2** | 20,000 | ... | 0.300 | ... | ... | ... | ... |
| **A2** | 20,000 | ... | 0.300 | ... | ... | ... | ... |

### BẢNG C: Kết quả Nghiên cứu Bóc tách ($\Delta = \text{Variant} - \text{Parent}$)
| Cặp so sánh | Mô hình cha (Parent) | Biến thể bóc tách (Variant) | Thành phần thay đổi duy nhất | Hiệu ứng mô hình cha | Hiệu ứng biến thể | Chênh lệch ghép cặp $\Delta$ | Ý nghĩa thống kê ($p$-value) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cặp 1** | H1 (Đầy đủ điều chuẩn) | A1 (Không điều chuẩn) | Loại bỏ Dropout (`rate = 0`) | ... | ... | ... | ... |
| **Cặp 2** | H2 (Đồ thị có ràng buộc) | A2 (Gỡ 1 ràng buộc) | Cho phép 1 cạnh cấm thời gian | ... | ... | ... | ... |
| **Cặp 3** | B0 (Bayesian Smoothing) | A3 (MLE thuần túy) | Bỏ BDeu prior, dùng đếm tần số | ... | ... | ... | ... |

---

## 5. Danh mục Sản phẩm nộp Cuối kỳ (Final Submission Checklist)

* [ ] File PDF bài báo cáo hoàn chỉnh được biên dịch từ LaTeX.
* [ ] Thư mục mã nguồn LaTeX (`.tex`, `.bib`, thư mục ảnh `figures/`).
* [ ] Mã nguồn Python / Notebook sạch, chạy tái lập từ đầu đến cuối trên môi trường sạch.
* [ ] File môi trường phụ thuộc (`requirements.txt` hoặc `environment.yml`).
* [ ] Thư mục artifacts chứa các file kết quả thực nghiệm dạng máy đọc được (`.json`, `.csv`).
* [ ] Bản khai báo minh bạch việc sử dụng công cụ AI (AI-Use Statement).
* [ ] Kiểm tra tỷ lệ trùng lặp và AI qua Turnitin trước khi nộp chính thức.
