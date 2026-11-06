# KHU VỰC BÁO CÁO GIỮA KỲ (MIDTERM DELIVERABLES)
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Thời gian làm bài:** 2 tuần | **Hình thức:** Báo cáo trực tuyến (Online)  
> **Thang điểm:** 10 điểm (4đ Foundations + 6đ Application)  
> **Cơ chế:** Duyệt đề tài (Accept / Reject)

---

## 1. Thành phần Nộp Bắt buộc Midterm

1. **Slide thuyết trình:** Đã hoàn thiện và lưu tại [Presentation_Midterm_Group_CausalML.pptx](./Presentation_Midterm_Group_CausalML.pptx).
   * Cấu trúc chuẩn hóa 13 slides bám sát 10 tiêu chí Rubric (Slide 42 & 43).
   * Thể hiện rõ: Bối cảnh, câu hỏi nhân quả, đồ thị Domain DAG B0, cấu trúc H1 và H2, 3 Bảng A, B, C, kết quả kiểm định DoWhy, phân tích ca thất bại và kịch bản demo trực tiếp 6 bước.
2. **File Notebook hoàn chỉnh:** File `.ipynb` tương thích Google Colab.
   * Có Mục lục (Table of Contents).
   * Đã chạy lưu sẵn toàn bộ output (Saved outputs), không có cell nào bị lỗi.
   * **Lưu ý:** Tất cả thành viên trong nhóm đều phải nộp file `.ipynb` này lên hệ thống trường.

---

## 2. Kịch bản Diễn tập Demo Trực tiếp (6 Bước Slide 59)

Trước buổi báo cáo Online, nhóm cần mở sẵn file Colab và diễn tập theo đúng thứ tự 6 bước:
* [ ] **Bước 1:** Nêu câu hỏi nhân quả trên tập UCI ID 697, trật tự thời gian biến số và 3 giả định Backdoor.
* [ ] **Bước 2:** Chiếu công thức định danh, bảng CPD của B0, và so sánh giá trị tương quan quan sát $P(Y \mid T)$ với can thiệp $P(Y \mid \text{do}(T))$.
* [ ] **Bước 3:** Mở khối mạng nơ-ron H1 (PyTorch MLP) và ma trận mặt nạ cạnh cấm $M$ của H2 (DAG-GNN).
* [ ] **Bước 4:** Thao tác đổi giá trị can thiệp (ví dụ từ $T=0$ sang $T=1$), chạy lại hàm tính rủi ro trực tiếp trước mặt Giảng viên.
* [ ] **Bước 5:** Trình chiếu 3 bảng số liệu: Bảng A (UCI), Bảng B (Synthetic Truth ATE = 0.30), Bảng C (Ablations) và kết quả 2 bài kiểm thử DoWhy Refutation.
* [ ] **Bước 6:** Trình bày 1 trường hợp thất bại / hạn chế của mô hình và sẵn sàng đối soát dòng code tạo ra con số đó.
