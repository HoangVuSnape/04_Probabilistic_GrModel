# HƯỚNG DẪN CHI TIẾT YÊU CẦU MIDTERM (THUYẾT TRÌNH & DEMO NOTEBOOK)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện (Nguyen Chi Thien, Ph.D.)  
> **Thời gian làm bài:** **2 tuần**  
> **Hình thức đánh giá:** Thuyết trình slide + Trình diễn trực tiếp mã nguồn trên Notebook (Google Colab / Jupyter) qua hình thức **Trực tuyến (Online)**  
> **Cơ chế duyệt đề tài:** **Accept / Reject** (Chấp thuận hoặc Từ chối đề tài dựa trên tính hợp lý của câu hỏi nhân quả và cấu trúc đồ thị)  
> **Quy mô nhóm:** 1 – 2 học viên (Group of 1–2 students)  
> **Quy tắc tính điểm nhóm:** $\text{Score}(\text{First Author}) - \text{Score}(\text{Co-Author}) = 1.0 \text{ điểm}$  
> **Yêu cầu nộp bài:** **Tất cả các thành viên trong nhóm đều phải nộp file `.ipynb` của nhóm lên hệ thống.**  
> **Tài liệu tham khảo:** [Google Drive Slide Hướng dẫn](https://drive.google.com/file/d/11vuIrPn783XorbLwZcXEGU3GWCNUEVdx/view?usp=sharing)  

---

## 1. Mục tiêu và Ý nghĩa của Đợt đánh giá Midterm

Đợt đánh giá Midterm kéo dài trong **2 tuần làm việc**, nhằm mục đích:
1. **Kiểm tra tính hợp lệ và phê duyệt đề tài (Accept/Reject):** Giảng viên thẩm định bài toán nhân quả, dữ liệu lựa chọn, tính hợp lý của biến can thiệp $T$, biến kết quả $Y$ và các biến đồng biến $Z$.
2. **Nắm vững nền tảng lý thuyết (Foundations - 4 điểm):** Hiểu rõ bản chất biểu diễn nhân quả, phân biệt rạch ròi giữa quan sát $P(Y \mid T=t)$ và can thiệp $P(Y \mid \text{do}(T=t))$.
3. **Năng lực triển khai thực nghiệm (Application - 6 điểm):** Chạy được mã nguồn trơn tru trên cả bài toán mô phỏng (Synthetic Benchmark) và bài toán thực tế (UCI Tabular).
4. **Trình diễn tương tác trực tiếp (Live Colab Demo):** Trình diễn 6 bước tuần tự, thay đổi giá trị can thiệp trực tiếp và đối soát dòng lệnh tạo ra kết quả.

---

## 2. Các nguyên tắc sống còn Giảng viên lưu ý trong Midterm

1. **"Đo can thiệp, không đo quan sát":**
   * Giảng viên đặc biệt nhấn mạnh: Tuyệt đối không đưa ra khuyến nghị chính sách dựa trên xác suất điều kiện quan sát $P(Y \mid T=t)$. Phải sử dụng công thức can thiệp đã qua phẫu thuật đồ thị (graph surgery / do-calculus) $P(Y \mid \text{do}(T=t))$.
2. **"Tính toán bất định — Kỳ vọng và Rủi ro":**
   * Kết quả báo cáo không được là một con số vô hồn. Phải tính được giá trị **Kỳ vọng (Expectation)**, **Rủi ro (Risk / Risk difference)** và **Độ bất định (Uncertainty / khoảng tin cậy / sai số chuẩn)**.
3. **"Chạy nhiều lần thử nghiệm — Chạy 1 lần là không đủ":**
   * Nếu chỉ chạy 1 lần rồi kết luận sẽ dẫn đến cái nhìn phiến diện và thiếu dữ kiện khoa học.
   * Bắt buộc phải thực hiện trên **ít nhất 5 hạt giống ngẫu nhiên (seeds $\ge 5$)** hoặc 5 lần chia dữ liệu ghép cặp để khẳng định tính ổn định.
4. **"Tìm node tìm grad vẽ ra -> DAG-graph (Gradient-based DAG)":**
   * Trong H2, phải dùng phương pháp tối ưu hóa liên tục (Continuous optimization - DAG-GNN / NOTEARS), dùng đạo hàm gradient để học ma trận trọng số $W$, áp dụng ma trận mặt nạ cấm cạnh (mask) rồi vẽ ra DAG.
   * **ChatGPT là không đủ:** Hỏi ChatGPT gợi ý đồ thị **không được tính** là thuật toán học cấu trúc.
5. **"Hướng tới giải bài toán thực tế chứ không phải biểu diễn công cụ":**
   * Mục tiêu tối thượng là trả lời câu hỏi thực tiễn: "Can thiệp gì để tạo ra thay đổi tích cực?".
   * Nếu tập dữ liệu bảng bị thiếu hoặc quá nghèo nàn: Hãy chủ động **thu thập thêm dữ liệu (crawl more data)** hoặc **kỹ nghệ thêm thuộc tính có ý nghĩa (add features to have more meaning)**.

---

## 3. Thang điểm Rubric chi tiết Midterm (10 điểm)

### PHẦN I: NỀN TẢNG LÝ THUYẾT (FOUNDATIONS — 4 ĐIỂM)

* **Tiêu chí 1: Biểu diễn nhân quả (Causal Representation) — 1.0 điểm [CLO1, CLO2]**
  * Định nghĩa rõ tập biến: Treatment $T$, Outcome $Y$, Confounders $Z$.
  * Vẽ đồ thị DAG cho B0, bảo vệ tính hợp lý của chiều mũi tên theo trật tự thời gian.
  * Viết công thức phân rã xác suất đồng thời (Factorization).
  * Nêu 3 giả định nhận diện: No unmeasured confounding, consistency, positivity ($0 < P(T=t \mid Z=z) < 1$).

* **Tiêu chí 2: Vai trò của Học tham số & Học sâu (Learning and DL Roles) — 2.0 điểm [CLO1, CLO2]**
  * Trình bày phương pháp làm mịn Bayesian smoothing (BDeu) cho B0.
  * Trình bày kiến trúc nơ-ron thay thế CPD trong H1 (đầu vào chỉ gồm parents của $Y$).
  * Trình bày cơ chế học đồ thị bằng gradient liên tục và ma trận mặt nạ cạnh cấm trong H2.
  * Phân tích rõ: Mạng nơ-ron khớp tốt dữ liệu quan sát không chứng minh được chiều nhân quả nếu thiếu giả định định danh.

* **Tiêu chí 3: Suy diễn quan sát và can thiệp (Inference) — 1.0 điểm [CLO1, CLO2]**
  * Phân biệt rạch ròi bằng công thức giữa $P(Y \mid T=t)$ và $P(Y \mid \text{do}(T=t))$.
  * Thể hiện cơ chế phân rã cắt cụt (Truncated Factorization): Xóa bỏ các cạnh hướng vào $T$ khi can thiệp.
  * Áp dụng tiêu chuẩn Backdoor để tìm tập điều chỉnh hợp lệ và biểu diễn công thức tính ATE.

---

### PHẦN II: ỨNG DỤNG VÀ THỰC NGHIỆM (APPLICATION — 6 ĐIỂM)

* **Tiêu chí 4: Lựa chọn bài toán & Dữ liệu UCI (Problem and UCI data) — 1.0 điểm [CLO3]**
  * Sử dụng dữ liệu bảng từ UCI (vd: Student Performance ID 320) hoặc dữ liệu thu thập thêm có kiểm chứng.
  * Đặt câu hỏi can thiệp nhân quả có ý nghĩa thực tế cao và có thể bảo vệ được.
  * Xử lý chống rò rỉ dữ liệu (loại bỏ biến kết quả tổng kết trước khi phân loại).
  * Kiểm tra tính bao phủ (treatment overlap check) để tránh các ô giá trị rỗng.

* **Tiêu chí 5: Cài đặt và tích hợp mã nguồn (Implementation) — 1.0 điểm [CLO3]**
  * Mã nguồn Python sạch, chạy thông suốt 100% trên Google Colab / Jupyter.
  * Có tối thiểu **3 mô hình B0, H1, H2** hoạt động trơn tru (khuyến khích mở rộng thêm H1+H2).
  * Tích hợp tường minh thư viện học sâu (PyTorch/TensorFlow) và thư viện nhân quả (`pgmpy`, `DoWhy`).

* **Tiêu chí 6: Giao thức đối sánh công bằng (Controlled Comparison) — 1.0 điểm [CLO3, CLO4]**
  * Dùng chung phân chia Train/Val/Test (vd: 70/15/15) cho mọi mô hình.
  * Chạy trên **ít nhất 5 hạt giống ngẫu nhiên (seeds $\ge 5$)**.
  * Tiền xử lý chỉ fit trên tập Train, tập Test giữ nguyên vẹn.
  * Báo cáo độ bất định (khoảng tin cậy), kỳ vọng, rủi ro và thời gian thực thi (runtime).

* **Tiêu chí 7: Kiểm chứng nhân quả (Causal Verification) — 1.0 điểm [CLO3, CLO4]**
  * Chạy bài toán mô phỏng phụ đạo (Synthetic Benchmark) và so sánh với giá trị chân lý lý thuyết ($ATE_{\text{true}} = 0.30$).
  * Chạy ít nhất 2 bài kiểm tra bác bỏ DoWhy: `placebo_treatment_refuter` và `data_subset_refuter`.
  * Thực hiện mô phỏng can thiệp với DoWhy GCM.

* **Tiêu chí 8: Nghiên cứu bóc tách (Ablation Study) — 1.0 điểm [CLO3, CLO4]**
  * Thực hiện đúng nguyên tắc "One change per variant" cho 3 biến thể:
    * **A1 từ H1:** Bỏ dropout / regularization.
    * **A2 từ H2:** Gỡ 1 ràng buộc cấm cạnh thời gian trong ma trận mask.
    * **A3 từ B0:** Dùng Maximum Likelihood Estimation thuần túy.
  * Báo cáo bảng chênh lệch ghép cặp có dấu $\Delta = \text{variant} - \text{parent}$.

* **Tiêu chí 9: Kết luận & Trả lời phản biện (Conclusions and Q&A) — 1.0 điểm [CLO4]**
  * Rút ra kết luận dựa trên số liệu thực nghiệm đo đạc, không khẳng định thiên vị rằng mô hình nơ-ron luôn tốt hơn.
  * Phân tích trung thực các trường hợp thất bại (lỗi ô xác suất 0 trong A3, lỗi học chu trình trong H2).
  * Trả lời tự tin, bảo vệ được phương pháp trước Hội đồng.

---

## 4. Kịch bản Trình diễn Colab Demo Online (6 bước bắt buộc)

Theo quy định tại **Slide 59**, khi bảo vệ trực tuyến trên lớp, nhóm phải trình chiếu Notebook và thực hiện theo đúng 6 bước:

```mermaid
sequenceDiagram
    autonumber
    actor Team as Nhóm sinh viên
    actor Lecturer as Giảng viên (Online)
    participant Colab as Màn hình Colab Notebook

    Team->>Lecturer: 1. Nêu câu hỏi nhân quả thực tế, biến số, đồ thị B0 & các giả định
    Team->>Colab: 2. Chiếu bảng CPD của B0, hiển thị giá trị P(Y|T) đối chiếu với P(Y|do(T))
    Team->>Colab: 3. Chỉ rõ khối mạng nơ-ron H1; mở ma trận mask ràng buộc miền & DAG học được của H2
    Team->>Colab: 4. Thao tác đổi giá trị can thiệp (T=0 -> T=1), chạy lại lệnh suy diễn trực tiếp
    Team->>Lecturer: Giải thích lý do vì sao can thiệp triệt tiêu các cạnh hướng vào T
    Team->>Colab: 5. Chiếu Bảng A (UCI), Bảng B (Synthetic ATE=0.30), Bảng C (Ablations) & DoWhy refutations
    Team->>Lecturer: 6. Phân tích 1 ca thất bại / kết quả rỗng & truy vết trực tiếp vào dòng code
```

---

## 5. Danh mục Sản phẩm nộp Midterm (Submission Checklist)

* [ ] File Slide thuyết trình: Định dạng `.pdf` hoặc `.pptx`.
* [ ] File Notebook: Định dạng `.ipynb` tương thích Google Colab, đã chạy lưu sẵn output tất cả các cell.
* [ ] **Tất cả các thành viên trong nhóm đã nộp file `.ipynb` lên hệ thống.**
* [ ] Đã ghi rõ tên First Author và Co-Author ở trang bìa Slide và đầu Notebook.
