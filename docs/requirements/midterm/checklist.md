# BẢNG CHECKLIST TỰ ĐÁNH GIÁ MIDTERM (10 ĐIỂM)
> Dành cho nhóm học viên tự rà soát trước buổi thuyết trình và duyệt đề tài **Midterm (Online)**.  
> **Thời gian thực hiện:** 2 tuần | **Quy mô nhóm:** 1–2 người | **Quy tắc điểm:** $\text{Score(First Author)} - \text{Score(Co-author)} = 1.0$

---

## I. THÔNG TIN HÀNH CHÍNH & ĐIỀU KIỆN TIÊN QUYẾT

- [ ] **Quy mô nhóm:** Đã đăng ký nhóm 1 hoặc 2 người.
- [ ] **Phân vai trò tác giả:** Đã xác định rõ **First Author** (phụ trách kiến trúc & thực nghiệm cốt lõi) và **Co-Author** (phụ trách phối hợp & phân tích).
- [ ] **Nộp bài:** **TẤT CẢ thành viên trong nhóm** đều đã nộp file `.ipynb` lên hệ thống.
- [ ] **Điều kiện duyệt đề tài (Accept/Reject):** Câu hỏi can thiệp nhân quả có ý nghĩa thực tế, tập dữ liệu có đủ biến số và trật tự thời gian được bảo vệ logic.

---

## II. CHECKLIST TIÊU CHÍ CHẤM ĐIỂM (10 ĐIỂM)

| Tiêu chí | Điểm | Nội dung kiểm tra chi tiết | Minh chứng trong Slide/Code | Đạt? |
| :--- | :---: | :--- | :--- | :---: |
| **1. Biểu diễn nhân quả** | **1.0** | - Định nghĩa từ điển biến số ($T, Y, Z$).<br/>- Vẽ đồ thị DAG cho B0, bảo vệ chiều mũi tên theo thời gian.<br/>- Viết công thức phân rã xác suất đồng thời.<br/>- Nêu 3 giả định: No unmeasured confounding, consistency, positivity. | Slide: ...<br/>Notebook Section: ... | [ ] |
| **2. Học tham số & Học sâu** | **2.0** | - Trình bày Bayesian smoothing (BDeu) cho B0.<br/>- Trình bày mạng nơ-ron cho H1 (đầu vào chỉ gồm parents của $Y$).<br/>- Trình bày học đồ thị bằng gradient liên tục (DAG-GNN/NOTEARS) kèm ma trận mặt nạ cấm cạnh trong H2.<br/>- Giải thích giới hạn: Mạng nơ-ron khớp dữ liệu không chứng minh được quan hệ nhân quả nếu thiếu định danh. | Slide: ...<br/>Notebook Section: ... | [ ] |
| **3. Suy diễn quan sát & can thiệp** | **1.0** | - Phân biệt công thức giữa $P(Y \mid T=t)$ và $P(Y \mid \text{do}(T=t))$.<br/>- Giải thích cơ chế phẫu thuật đồ thị (xóa cạnh đến $T$).<br/>- Trình bày tiêu chuẩn Backdoor và công thức tính ATE. | Slide: ...<br/>Notebook Section: ... | [ ] |
| **4. Lựa chọn bài toán & Dữ liệu** | **1.0** | - Sử dụng dữ liệu bảng UCI (hoặc dữ liệu bổ sung / crawl thêm nếu thiếu dữ liệu).<br/>- Định nghĩa câu hỏi can thiệp thực tế, hướng tới giải bài toán.<br/>- Loại bỏ biến rò rỉ (data leakage).<br/>- Kiểm tra tính bao phủ (treatment overlap) để tránh ô rỗng. | Slide: ...<br/>Notebook Section: ... | [ ] |
| **5. Cài đặt mã nguồn** | **1.0** | - Mã nguồn chạy thông suốt 100% trên Colab, không báo lỗi.<br/>- Triển khai tối thiểu 3 mô hình B0, H1, H2 (khuyến khích có thêm H1+H2).<br/>- Tích hợp PyTorch/TensorFlow, pgmpy, DoWhy. | Notebook: toàn bộ tế bào có saved output | [ ] |
| **6. Đối sánh công bằng** | **1.0** | - Dùng chung split Train/Val/Test (vd: 70/15/15).<br/>- Chạy trên **ít nhất 5 hạt giống ngẫu nhiên (seeds $\ge 5$)**.<br/>- Tiền xử lý chỉ fit trên tập Train.<br/>- Báo cáo Kỳ vọng, Rủi ro, Độ bất định (khoảng tin cậy) và thời gian chạy. | Slide (Bảng A): ...<br/>Notebook Section: ... | [ ] |
| **7. Kiểm chứng nhân quả** | **1.0** | - Chạy bài toán Synthetic Tutoring, so sánh với chân lý ($ATE=0.30$).<br/>- Chạy 2 kiểm định bác bỏ DoWhy: Placebo refuter & Data subset refuter.<br/>- Mô phỏng can thiệp bằng DoWhy GCM. | Slide (Bảng B): ...<br/>Notebook Section: ... | [ ] |
| **8. Nghiên cứu bóc tách** | **1.0** | - Thực hiện đúng 3 phép bóc tách "One change only":<br/>  * A1 từ H1: Bỏ dropout / regularization.<br/>  * A2 từ H2: Gỡ 1 ràng buộc cấm cạnh thời gian.<br/>  * A3 từ B0: Dùng MLE thay vì Bayesian smoothing.<br/>- Báo cáo bảng chênh lệch ghép cặp có dấu $\Delta$. | Slide (Bảng C): ...<br/>Notebook Section: ... | [ ] |
| **9. Kết luận & Trả lời Q&A** | **1.0** | - Đưa ra kết luận khách quan từ dữ liệu, không thiên lệch mô hình nơ-ron.<br/>- Báo cáo trung thực các ca thất bại / kết quả rỗng.<br/>- Mọi thành viên hiểu sâu mã nguồn và đối soát được số liệu với code. | Trực tiếp trong buổi bảo vệ Online | [ ] |

---

## III. CHECKLIST 6 BƯỚC DEMO TRỰC TIẾP TRÊN GOOGLE COLAB (SLIDE 59)

- [ ] **Bước 1:** Trình bày rõ câu hỏi can thiệp thực tế, biến $T, Y, Z$, đồ thị B0 và giả định Backdoor.
- [ ] **Bước 2:** Chiếu công thức định danh, bảng CPD của B0, và so sánh xác suất quan sát $P(Y \mid T)$ với xác suất can thiệp $P(Y \mid \text{do}(T))$.
- [ ] **Bước 3:** Mở khối mã nguồn H1 (chỉ rõ vị trí mạng nơ-ron thay thế CPD) và H2 (chỉ rõ ma trận mặt nạ mask và đồ thị học được từ gradient).
- [ ] **Bước 4:** Thao tác đổi trực tiếp $T=0 \rightarrow T=1$ trên Notebook, chạy lại lệnh tính rủi ro và giải thích nguyên lý cắt bỏ cạnh đi vào $T$.
- [ ] **Bước 5:** Trình chiếu Bảng A (so sánh B0/H1/H2 trên UCI), Bảng B (đối chiếu chân lý Synthetic 0.30), Bảng C (bóc tách A1–A3) và log của 2 phép kiểm thử DoWhy.
- [ ] **Bước 6:** Phân tích 1 trường hợp thất bại / hạn chế mô hình và sẵn sàng chỉ ra dòng code tạo ra kết quả.
