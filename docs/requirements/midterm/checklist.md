# BẢNG CHECKLIST TỰ ĐÁNH GIÁ MIDTERM (10 ĐIỂM)
> Dành cho nhóm học viên tự rà soát trước buổi thuyết trình và duyệt đề tài **Midterm (Online)**.  
> **Thời gian thực hiện:** 2 tuần | **Quy mô nhóm:** 1–2 người | **Quy tắc điểm:** $\text{Score(First Author)} - \text{Score(Co-author)} = 1.0$  
> **Tài liệu tham chiếu:** Slide 39–44, 51, 53–75, 76–77, 79 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

---

## I. THÔNG TIN HÀNH CHÍNH & ĐIỀU KIỆN TIÊN QUYẾT

- [ ] **Quy mô nhóm:** Đã đăng ký nhóm 1 hoặc 2 người.
- [ ] **Phân vai trò tác giả:** Đã xác định rõ **First Author** (chịu trách nhiệm chính về kiến trúc & thực nghiệm cốt lõi) và **Co-Author** (chịu trách nhiệm phối hợp triển khai, phân tích dữ liệu & kiểm thử).
- [ ] **Nộp bài:** **TẤT CẢ thành viên trong nhóm** đều đã nộp file `.ipynb` của nhóm lên hệ thống trước hạn.
- [ ] **Điều kiện duyệt đề tài (Accept / Reject):** Câu hỏi can thiệp nhân quả có ý nghĩa thực tế trên 01 bộ dữ liệu bảng UCI, trật tự thời gian được bảo vệ logic từ tài liệu khoa học.

---

## II. CHECKLIST TIÊU CHÍ CHẤM ĐIỂM RUBRIC (10 ĐIỂM — SLIDE 42 & 43)

| Tiêu chí | Điểm | Nội dung kiểm tra chi tiết theo Slide 80 | Vị trí minh chứng (Slide / Notebook) | Đạt? |
| :--- | :---: | :--- | :--- | :---: |
| **1. Biểu diễn nhân quả** | **1.0** | - Định nghĩa rõ câu hỏi can thiệp (dân số, treatment $T$, outcome $Y$, thời điểm can thiệp).<br/>- Bảng từ điển biến số (tên, mã hóa, thời điểm đo lường, vai trò causal, % missing).<br/>- Vẽ đồ thị DAG cho B0, bảo vệ tính hợp lý của chiều mũi tên theo thời gian.<br/>- Viết công thức phân rã xác suất đồng thời (Factorization).<br/>- Tuyên bố 3 giả định nhận diện: No unmeasured confounding, consistency, positivity. | Slide: ...<br/>Notebook Section: Setup & DAG | [ ] |
| **2. Vai trò học tham số & học sâu** | **2.0** | - Trình bày Bayesian smoothing (BDeu prior) cho B0.<br/>- Trình bày mạng nơ-ron cho H1 (đầu vào chỉ gồm parents của $Y$, hàm mất mát BCE, tạo bảng CPD chuẩn hóa).<br/>- Trình bày học đồ thị bằng gradient liên tục (DAG-GNN/NOTEARS) kèm ma trận mặt nạ cấm cạnh trong H2.<br/>- Phân tích giới hạn: Mạng nơ-ron khớp tốt dữ liệu quan sát không chứng minh được quan hệ nhân quả nếu thiếu định danh. | Slide: ...<br/>Notebook Section: H1 & H2 | [ ] |
| **3. Suy diễn quan sát & can thiệp** | **1.0** | - Phân biệt rạch ròi bằng công thức giữa $P(Y \mid T=t)$ và $P(Y \mid \text{do}(T=t))$.<br/>- Giải thích cơ chế phẫu thuật đồ thị (xóa các cạnh đi vào $T$ khi can thiệp).<br/>- Trình bày tiêu chuẩn Backdoor và công thức tính ATE chuẩn hóa trên tập tham chiếu $C_{\text{test}}$. | Slide: ...<br/>Notebook Section: Inference | [ ] |
| **4. Lựa chọn bài toán & Dữ liệu UCI** | **1.0** | - Sử dụng 01 bộ dữ liệu bảng UCI (vd: Student Performance ID 320 hoặc Student Dropout ID 697).<br/>- Lưu đầy đủ metadata: URL, checksum file thô, license.<br/>- Báo cáo lưu đồ lọc dữ liệu (eligibility flow) trước và sau lọc; xử lý chống rò rỉ nhãn kết quả.<br/>- Kiểm toán trật tự thời gian và kiểm tra độ bao phủ can thiệp (treatment overlap), chỉ ra các nhóm thưa. | Slide: ...<br/>Notebook Section: Data Audit | [ ] |
| **5. Cài đặt mã nguồn tái lập** | **1.0** | - Mã nguồn chạy thông suốt 100% trên Colab / Jupyter từ đầu đến cuối không lỗi (fresh runtime restart).<br/>- Cài đặt tối thiểu 3 mô hình B0, H1, H2 (khuyến khích mở rộng tùy chọn H1+H2).<br/>- Dùng một khối cấu hình duy nhất (`ExperimentConfig`); lưu giữ checkpoint, ma trận kề và artifacts máy đọc được. | Notebook Colab đính kèm đầy đủ outputs | [ ] |
| **6. Đối sánh dự báo ngoài mẫu (Bảng A)** | **1.0** | - Dùng chung phân chia Train/Val/Test (vd: 70/15/15) và chạy trên **ít nhất 5 paired runs**.<br/>- Tiền xử lý chỉ fit trên Train; Val chọn ngưỡng/stopping; Test đóng băng đánh giá.<br/>- Báo cáo Bảng A (Slide 74): AUROC, Macro-F1, Brier Score/Log Loss; chênh lệch ghép cặp có dấu ($H_1 - B_0$, $H_2 - B_0$). | Slide: Bảng A<br/>Notebook Section: Table A | [ ] |
| **7. Đối sánh can thiệp chuẩn hóa (Bảng B)** | **1.0** | - Chuẩn hóa nguy cơ trên cùng tập hiệp biến nền kiểm thử $C_{\text{test}}$ cho B0, H1, H2.<br/>- Báo cáo Bảng B (Slide 75): $\text{risk}(T=0)$, $\text{risk}(T=1)$, $\widehat{\text{ATE}}$ và khoảng tin cậy 95% kèm phương pháp tính.<br/>- Tuyệt đối không dùng nhãn thực tế $Y_{\text{test}}$ để tính hiệu ứng can thiệp! | Slide: Bảng B<br/>Notebook Section: Table B | [ ] |
| **8. Tính vững bác bỏ & Chi phí (Bảng C)** | **1.0** | - Chạy **ít nhất 2 bài kiểm tra bác bỏ DoWhy**: `placebo_treatment_refuter` và `data_subset_refuter`.<br/>- Báo cáo Bảng C (Slide 75): Kết quả kiểm định bác bỏ, độ biến thiên hiệu ứng qua 5 runs (paired effect spread), thời gian chạy (fit/query runtime) và cấu hình phần cứng. | Slide: Bảng C<br/>Notebook Section: Table C | [ ] |
| **9. Kết luận & Bảo vệ Q&A** | **1.0** | - Đưa ra kết luận khách quan từ dữ liệu đo đạc, không thiên vị Deep Learning.<br/>- Phân tích trung thực ít nhất một ca thất bại / kết quả rỗng / ô xác suất 0 theo Slide 79.<br/>- Hiểu sâu mã nguồn, sẵn sàng truy vết từ bảng số liệu về dòng lệnh tạo ra kết quả. | Buổi bảo vệ trực tuyến (Online) | [ ] |

> [!NOTE]
> **Phần mở rộng tùy chọn (Optional Extensions):**
> - [ ] *(Tùy chọn)* Bài toán mô phỏng phụ đạo (Synthetic Tutoring Benchmark, $ATE=0.30$) và kiểm định DoWhy GCM (Slide 72–73).
> - [ ] *(Tùy chọn)* Nghiên cứu bóc tách 3 biến thể A1, A2, A3 (Slide 64–67) và mô hình kết hợp H1+H2 (Slide 80).
> *(Các phần này nếu thực hiện sẽ được trình bày ở phụ lục slide/notebook để minh họa sâu thêm).*

---

## III. CHECKLIST 6 BƯỚC DEMO TRỰC TIẾP TRÊN GOOGLE COLAB (SLIDE 77)

- [ ] **Bước 1 (Câu hỏi nhân quả & Bối cảnh):** Nêu câu hỏi can thiệp UCI, trật tự thời gian, đồ thị và các giả định; chỉ rõ tập tham chiếu $C_{\text{test}}$ và hai mức can thiệp $T=0, T=1$.
- [ ] **Bước 2 (Minh bạch phân định dữ liệu):** Giải thích rõ TRAIN (fit mô hình/tiền xử lý), VALIDATION (chọn stopping/ngưỡng đồ thị), TEST (đánh giá dự báo ngoài mẫu & chuẩn hóa can thiệp).
- [ ] **Bước 3 (Kiến trúc H1 & H2):** Chỉ rõ vị trí thay thế CPD nơ-ron của $Y$ trong H1; mở ma trận trọng số kề $W$, mặt nạ cấm cạnh $M$ và đồ thị DAG học được của H2 kèm kết quả DoWhy identify.
- [ ] **Bước 4 (Tính toán can thiệp trực tiếp):** Chọn vài profile cá nhân $C_i$ từ tập Test, hiển thị $q_m(0, C_i)$ và $q_m(1, C_i)$; chạy lệnh lấy trung bình trên toàn bộ $C_{\text{test}}$ để tính $\text{risk}_m(t)$ và $\text{ATE}_m$; giải thích ý nghĩa cắt bỏ các cạnh vào $T$.
- [ ] **Bước 5 (Trình chiếu Bộ 3 Bảng Kết quả):** Trình chiếu Bảng A (so sánh dự báo TEST), Bảng B (so sánh hiệu ứng can thiệp & 95% CI), Bảng C (độ ổn định qua 5 runs, 2 kiểm định bác bỏ DoWhy và thời gian chạy).
- [ ] **Bước 6 (Phân tích lỗi & Truy vết mã nguồn):** Trình bày trung thực một trường hợp thất bại / kết quả rỗng / giới hạn nhân quả (theo Slide 79) và truy vết ngay lập tức từ bảng số liệu về file config và dòng code tương ứng.
