# 04. KẾ HOẠCH HÀNH ĐỘNG 14 NGÀY & LỘ TRÌNH CÁN ĐÍCH (EXECUTION PLAN & MILESTONES)
> **Dự án:** Causal Bayesian Networks with Deep Learning for Educational Interventions  
> **Thời gian Midterm:** **2 tuần (14 ngày)** | **Hình thức:** Báo cáo trực tuyến (Online)  
> **Phân chia vai trò:** Nhóm 1–2 người (Quy tắc điểm: $\text{First Author} - \text{Co-Author} = 1.0$)

---

## 1. Phân bổ Vai trò Nhóm 2 Người (Team Roles)

Để đạt hiệu quả tối đa và đảm bảo tính công bằng theo quy chế đánh giá:

* **First Author (Tác giả chính — Chịu trách nhiệm 60% khối lượng):**
  * Định nghĩa bài toán nhân quả, xây dựng đồ thị Domain DAG và chứng minh nhận diện Backdoor.
  * Thiết kế và huấn luyện mạng nơ-ron H1 (PyTorch) và thuật toán học đồ thị H2 (DAG-GNN / NOTEARS có mask).
  * Tích hợp suy diễn can thiệp `DoWhy`, viết các kiểm định bác bỏ nhân quả (Refutation checks).
  * Trình bày phần lý thuyết và thực hiện thao tác tương tác chính trong buổi Demo online.
* **Co-Author (Đồng tác giả — Chịu trách nhiệm 40% khối lượng):**
  * Thu thập dữ liệu qua `ucimlrepo`, lập trình pipeline tiền xử lý (rời rạc hóa, mã hóa, chia 5 split).
  * Triển khai bài toán mô phỏng Synthetic Benchmark và 3 biến thể bóc tách A1, A2, A3.
  * Quản lý kết xuất số liệu, điền dữ liệu vào 3 Bảng A, B, C; thiết kế Slide thuyết trình.
  * Chuẩn bị khung sườn LaTeX và tài liệu báo cáo hướng tới đợt thi Cuối kỳ.

---

## 2. Lịch trình Hành động Chi tiết 14 Ngày cho Midterm

| Giai đoạn | Ngày | Công việc | Thời lượng |
|---|---:|---|---:|
| 1. Nền tảng | 1 | Chốt bài toán và thẩm định dữ liệu UCI 697 | Mốc |
| 1. Nền tảng | 1-2 | Lập trình pipeline nạp và làm sạch dữ liệu | 2 ngày |
| 1. Nền tảng | 3-4 | Cài đặt Synthetic Benchmark, kiểm tra true ATE = 0.30 | 2 ngày |
| 2. Baseline và mô hình lai | 5-6 | Cài đặt B0 với pgmpy và DoWhy | 2 ngày |
| 2. Baseline và mô hình lai | 7-9 | Cài đặt H1, A1 và A3 | 3 ngày |
| 3. Học cấu trúc và đối soát | 10-11 | Cài đặt H2, domain mask và A2 | 2 ngày |
| 3. Học cấu trúc và đối soát | 12 | Chạy 5 seeds, xuất Bảng A/B/C và refutation | 1 ngày |
| 4. Hoàn thiện và diễn tập | 13 | Thiết kế slide, rà soát notebook | 1 ngày |
| 4. Hoàn thiện và diễn tập | 14 | Diễn tập demo 6 bước và nộp bài | Mốc |

> Dùng bảng Markdown thay cho Mermaid Gantt để tài liệu hiển thị ổn định trong
> VS Code và không phụ thuộc phiên bản Mermaid của trình xem.

### Chi tiết từng mốc công việc:

#### 🟢 GIAI ĐOẠN 1: DỮ LIỆU & BÀI TOÁN MÔ PHỎNG (NGÀY 1 – 4)
* **Ngày 1 (Duyệt đề tài):** Nộp bản đề xuất đề tài (Causal Question, Variable Dictionary, Domain DAG) cho Thầy để nhận đánh giá **ACCEPT**.
* **Ngày 2 (Data Pipeline):** Viết script nạp dữ liệu UCI ID 697, lọc 794 sinh viên `Enrolled`, kiểm tra missing values, mã hóa nhãn nhị phân cho $T$ và $Y$.
* **Ngày 3 (Kiểm tra Overlap & Chia Splits):** Kiểm tra ma trận tần số giữa $T$ và các nhóm $Z$; chia tập Train/Val/Test (70/15/15) trên 5 seeds: `[42, 123, 456, 789, 999]`.
* **Ngày 4 (Synthetic Benchmark):** Hoàn thiện khối code sinh 20,000 mẫu mô phỏng bài toán phụ đạo (Tutoring). Xác nhận code tính ra đúng $P(Y=1 \mid \text{do}(T=0)) = 0.30$, $P(Y=1 \mid \text{do}(T=1)) = 0.60$, $\text{ATE} = 0.30$.

#### 🟡 GIAI ĐOẠN 2: BASELINE B0 & MÔ HÌNH LAI H1 (NGÀY 5 – 9)
* **Ngày 5 – 6 (Mô hình B0):** Xây dựng mạng Bayes `pgmpy` cho B0, khớp tham số với `BayesianEstimator(BDeu, ESS=5)`. Kết nối `DoWhy` để nhận diện Backdoor và tính ATE trên cả dữ liệu Synthetic và UCI.
* **Ngày 7 – 8 (Mô hình H1):** Viết mạng PyTorch MLP học phân phối $P_\theta(Y \mid \text{Parents}(Y))$, huấn luyện với BCE loss, early stopping trên validation set. Trích xuất bảng xác suất nơ-ron đưa vào BN.
* **Ngày 9 (Bóc tách A1 & A3):**
  * Chạy biến thể **A1:** Huấn luyện lại H1 khi bỏ Dropout (`p=0.0`).
  * Chạy biến thể **A3:** Khớp lại B0 với `MaximumLikelihoodEstimator`.

#### 🟠 GIAI ĐOẠN 3: HỌC CẤU TRÚC H2 & KIỂM ĐỊNH BÁC BỎ (NGÀY 10 – 12)
* **Ngày 10 – 11 (Mô hình H2 & Bóc tách A2):**
  * Cài đặt thuật toán học đồ thị vi phân (DAG-GNN / NOTEARS).
  * Áp đặt ma trận mặt nạ cạnh cấm (Domain Mask $M$) cấm các cạnh vi phạm thời gian ($Y \rightarrow T, Y \rightarrow Z, T \rightarrow Z$).
  * Chạy biến thể **A2:** Gỡ bỏ 1 cạnh cấm ($M_{Y, Z_1} = 1$), huấn luyện lại để kiểm tra độ nhạy cảm của tri thức miền.
* **Ngày 12 (Tổng hợp 5 Seeds & DoWhy Refutation):**
  * Thu thập kết quả 5 seeds, tính giá trị trung bình $\pm$ độ lệch chuẩn cho **Bảng A** (UCI), **Bảng B** (Synthetic) và **Bảng C** (Ablations).
  * Chạy 2 kiểm định bác bỏ: `placebo_treatment_refuter` và `data_subset_refuter`.

#### 🔴 GIAI ĐOẠN 4: SLIDE, DEMO & NỘP BÀI (NGÀY 13 – 14)
* **Ngày 13 (Soạn thảo Slide & Colab):**
  * Làm Slide thuyết trình đầy đủ 10 phần theo Rubric (4đ Nền tảng + 6đ Ứng dụng).
  * Dọn dẹp Google Colab Notebook sạch đẹp: Có Table of Contents, chạy lưu sẵn toàn bộ output, không còn bất kỳ lỗi nào.
* **Ngày 14 (Diễn tập & Nộp bài):**
  * Diễn tập kịch bản **6 bước Demo trực tuyến** theo quy định tại Slide 59.
  * **Tất cả các thành viên trong nhóm nộp file `.ipynb` lên hệ thống.**

---

## 3. Lộ trình Tiếp nối Cán đích Cuối kỳ (Final Roadmap)

Sau khi hoàn thành Midterm, nhóm sẽ chuyển trọng tâm sang đợt Final:
1. **Chuyển dịch nội dung sang LaTeX:** Sử dụng khung sườn tại `docs/requirements/final/report_template_structure.md`, điền các phần phân tích chuyên sâu từ mục 2.1 đến 2.9.
2. **Kiểm tra tỷ lệ AI qua Turnitin:** Rà soát câu từ, tự viết diễn giải các bảng số liệu khoa học, đảm bảo bài viết vượt qua kiểm duyệt Turnitin AI Writing Check một cách an toàn.
3. **Đóng gói mã nguồn tái lập (Reproducible Package):** Đóng gói file môi trường `requirements.txt`, lưu trữ các file artifacts (`.json`, `.csv`) để người chấm có thể tái lập kết quả chỉ bằng 1 câu lệnh.
