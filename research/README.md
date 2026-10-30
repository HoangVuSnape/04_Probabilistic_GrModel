# THƯ MỤC NGHIÊN CỨU & NHẬT KÝ ĐỀ TÀI (RESEARCH & EXPERIMENT LOGS)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Giảng viên hướng dẫn:** TS. Nguyễn Chí Thiện  
> **Bộ dữ liệu đã chốt:** **Predict Students' Dropout and Academic Success** (UCI ID: **697**)  
> **Mục tiêu nghiên cứu:** Đánh giá hiệu quả can thiệp của chính sách học bổng / hỗ trợ tài chính lên sự thành công học tập của sinh viên bằng Mạng Bayes Nhân quả Lai ghép Học sâu.

---

## 1. Mục đích và Ý nghĩa của Thư mục `research/`

Thư mục `research/` là **trung tâm quản trị tri thức và nhật ký thực nghiệm (Single Source of Truth)** cho toàn bộ quá trình nghiên cứu từ giai đoạn khởi động, duyệt đề tài Midterm (Accept/Reject), cho đến hoàn thiện bài báo khoa học Cuối kỳ (Final).

Toàn bộ các quyết định kỹ thuật, công thức toán học, thiết kế thực nghiệm và nhật ký từng lần chạy đều được lưu trữ có hệ thống tại đây để phục vụ:
1. Soạn thảo Slide và chuẩn bị kịch bản 6 bước Demo trực tuyến cho đợt **Midterm (2 tuần)**.
2. Biên dịch bài báo khoa học chuẩn LaTeX và đối soát số liệu cho đợt **Final (Turnitin Check)**.
3. Đảm bảo tính minh bạch, tái lập (reproducibility) tuyệt đối cho người chấm.

---

## 2. Quy tắc Đặt tên & Cấu trúc Tài liệu bên trong

Tất cả các tài liệu nghiên cứu trong thư mục này được tổ chức theo quy chuẩn đánh số thứ tự **`01_<Tên_Nội_Dung>.md`**:

```text
research/
├── README.md                      <-- File này: Tổng quan thư mục và điều hướng
├── 01_problem_formulation.md      <-- Đặt vấn đề nhân quả, câu hỏi can thiệp, đồ thị Causal DAG và giả định Backdoor
├── 02_dataset_specification.md    <-- Đặc tả chi tiết bộ dữ liệu UCI ID 697, kiểm toán dữ liệu và kiểm tra tính bao phủ (Overlap)
├── 03_cbn_and_dl_architecture.md  <-- Thiết kế chi tiết 3 mô hình B0, H1, H2, mở rộng H1+H2 và 3 biến thể bóc tách A1–A3
├── 04_execution_plan_and_timeline.md <-- Kế hoạch hành động 14 ngày cho Midterm và lộ trình cán đích Final
├── 05_experiment_logs.md          <-- Nhật ký thực nghiệm (Tracking logs, kết quả qua các seeds, các lỗi và cách khắc phục)
├── 06_literature_review_and_limitations.md <-- Tổng quan học thuật, thảo luận từ Kaggle S4E6, hạn chế dữ liệu và các hướng tiếp cận mới
├── 07_curated_15_research_papers.md <-- Danh mục 15 bài báo khoa học chuẩn mực theo đúng Rule môn học (kèm link arXiv/DOI)
├── 08_kaggle_s4e6_data_deep_dive.md <-- Đào sâu synthetic data Kaggle S4E6, discussion, ưu/nhược điểm và khuyến nghị sử dụng
├── 09_requirements_aligned_research_approach.md <-- Định hướng nghiên cứu bám rubric Midterm/Final và kế hoạch triển khai
└── 10_eda_plan_and_report_uci697_kaggle_s4e6.md <-- Kế hoạch và báo cáo khảo sát dữ liệu UCI/Kaggle
```

---

## 3. Tóm tắt Quyết định Đề tài đã Chốt

### 3.1. Dữ liệu thực nghiệm chính
* **Tên bộ dữ liệu:** Predict Students' Dropout and Academic Success (UCI ID: 697).
* **Quy mô:** 4,424 sinh viên đại học, 36 biến đặc trưng kinh tế - xã hội, nhân khẩu học và kết quả học tập.
* **Lý do lựa chọn:**
  * Thuộc cùng miền bài toán Giáo dục (Education) như bài giảng mẫu của Thầy, nhưng quy mô lớn hơn, thuộc tính phong phú hơn nhiều so với bộ Student Performance (ID 320 cũ).
  * Trật tự thời gian giữa các biến số rõ ràng, phân định rạch ròi trước và sau can thiệp.

### 3.2. Bài toán Nhân quả Cốt lõi
* **Biến can thiệp ($T$):** Sinh viên có nhận học bổng / hỗ trợ tài chính hay không (`Scholarship holder`: $0 = \text{Không}, 1 = \text{Có}$).
* **Biến kết quả ($Y$):** Tốt nghiệp đại học thành công (`Target`: $1 = \text{Graduate}, 0 = \text{Dropout}$).
* **Biến đồng biến nền ($Z$):** Đo trước khi nhập học / trước khi xét học bổng: Điểm xét tuyển đầu vào (`Admission grade`), trình độ học vấn của bố mẹ, tình trạng nợ học phí (`Debtor`), học phí đóng đúng hạn (`Tuition fees up to date`), tuổi khi nhập học (`Age at enrollment`).
* **Câu hỏi nghiên cứu:**
  $$\text{ATE} = \mathbb{E}[Y \mid \text{do}(T=1)] - \mathbb{E}[Y \mid \text{do}(T=0)]$$
  *"Nếu nhà trường can thiệp áp đặt chính sách cấp học bổng cho toàn bộ sinh viên, xác suất tốt nghiệp thành công thay đổi bao nhiêu so với khi không cấp học bổng?"*

### 3.3. Hệ thống mô hình
1. **B0:** Mạng Bayes cổ điển với Domain DAG + Bayesian smoothing (BDeu prior).
2. **H1:** Cố định DAG B0, thay CPD của $Y$ bằng mạng nơ-ron MLP sâu.
3. **H2:** Học đồ thị bằng Gradient liên tục (DAG-GNN / NOTEARS) với ma trận mặt nạ cạnh cấm (Domain Mask $M$).
4. **H1+H2 (Mở rộng):** Kết hợp cấu trúc học được từ H2 và cơ chế tham số nơ-ron từ H1.
5. **A1–A3:** Ba biến thể bóc tách (Bỏ dropout, gỡ 1 ràng buộc cấm cạnh, dùng MLE).

---

## 4. Điều hướng nhanh các tài liệu

* 🎯 [01. Đặt vấn đề & Thiết kế Đồ thị Nhân quả (Problem Formulation)](./01_problem_formulation.md)
* 📊 [02. Đặc tả Chi tiết Bộ dữ liệu UCI ID 697 (Dataset Specification)](./02_dataset_specification.md)
* 🧠 [03. Kiến trúc Kỹ thuật B0, H1, H2, A1–A3 (Model Architecture)](./03_cbn_and_dl_architecture.md)
* 📅 [04. Kế hoạch Hành động 14 ngày & Lộ trình Nộp bài (Execution Plan)](./04_execution_plan_and_timeline.md)
* 📝 [05. Nhật ký Thực nghiệm & Đối soát Kết quả (Experiment Logs)](./05_experiment_logs.md)
* 📚 [06. Tổng quan Học thuật, Thảo luận & Hạn chế Dữ liệu (Literature Review)](./06_literature_review_and_limitations.md)
* 📑 [07. Danh mục 15 Bài báo Khoa học Chuẩn mực (Curated Papers)](./07_curated_15_research_papers.md)
* 🔎 [08. Đào sâu Kaggle S4E6 và Synthetic Data](./08_kaggle_s4e6_data_deep_dive.md)
* 🎯 [09. Định hướng bám Requirements Midterm/Final](./09_requirements_aligned_research_approach.md)
* 📊 [10. Kế hoạch & Báo cáo EDA UCI/Kaggle](./10_eda_plan_and_report_uci697_kaggle_s4e6.md)
