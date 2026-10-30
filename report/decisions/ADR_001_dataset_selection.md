# ADR-001: LỰA CHỌN BỘ DỮ LIỆU THỰC NGHIỆM CHÍNH (UCI ID: 697)
> **Trạng thái:** ACCEPTED (ĐÃ CHẤP THUẬN)  
> **Ngày phê duyệt:** 2026-10-29  
> **Người đề xuất:** Nhóm nghiên cứu  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  

---

## 1. Bối cảnh (Context)
* Slide môn học yêu cầu nghiên cứu phải sử dụng một bộ dữ liệu dạng bảng công khai từ **UCI Machine Learning Repository** (Slide 15).
* Giáo trình giảng dạy đã lấy bài toán Giáo dục (*Student Performance*, UCI ID: 320) làm ví dụ mẫu (Slide 16, 17, 37, 40).
* Tuy nhiên, bộ dữ liệu UCI ID 320 chỉ có 395 hoặc 649 học sinh phổ thông, số lượng mẫu tương đối hạn chế, dễ dẫn đến hiện tượng ô tần số thưa (sparse cells) khi chia tập và rời rạc hóa.
* Nhóm cần một bộ dữ liệu thuộc miền giáo dục (đảm bảo bám sát kiến thức trên lớp), nhưng có quy mô lớn hơn, thuộc tính đa dạng hơn và có tính thời gian rõ ràng hơn để phục vụ cho cả báo cáo Cuối kỳ.

---

## 2. Các phương án xem xét (Options Considered)
1. **Phương án A: Sử dụng bộ dữ liệu cũ Student Performance (UCI ID: 320)**
   * *Ưu điểm:* Khớp 100% với tên gọi trong slide của Thầy.
   * *Nhược điểm:* Cỡ mẫu nhỏ (395 dòng ở môn Toán), chỉ có 33 thuộc tính; dễ gặp lỗi xác suất 0 khi khớp bảng phân phối rời rạc; tính mới học thuật không cao.
2. **Phương án B: Sử dụng bộ dữ liệu Predict Students' Dropout and Academic Success (UCI ID: 697) [ĐƯỢC CHỌN]**
   * *Ưu điểm:* 
     * Quy mô lớn: **4,424 dòng**, **36 thuộc tính**.
     * Được công bố năm 2021 tại Bồ Đào Nha, dữ liệu sạch, không khuyết thiếu.
     * Có biến can thiệp chính sách rất mạnh: Sinh viên có nhận học bổng / hỗ trợ tài chính hay không (`Scholarship holder`).
     * Phân tách rạch ròi giữa biến nền tảng trước nhập học (điểm tuyển sinh, học vấn cha mẹ, tình trạng nợ học phí) và kết quả cuối khóa (tốt nghiệp vs bỏ học).
3. **Phương án C: Chuyển sang miền Y tế hoặc Kinh tế (UCI ID: 296 hoặc ID: 2)**
   * *Ưu điểm:* Cỡ mẫu cực lớn.
   * *Nhược điểm:* Phức tạp trong việc thiết lập trật tự đồ thị Domain DAG tiên nghiệm, lệch so với ví dụ giáo dục được hướng dẫn trên slide.

---

## 3. Quyết định (Decision)
Nhóm chính thức quyết định lựa chọn **Phương án B: Bộ dữ liệu Predict Students' Dropout and Academic Success (UCI ID: 697)**.

* **Quy trình xử lý dữ liệu:**
  * Lọc bỏ 794 sinh viên có trạng thái `Enrolled` (vẫn đang học, chưa rõ kết quả) để thu được tập mẫu hoàn tất $N = 3,630$ sinh viên.
  * Nhãn $Y$: Nhị phân rõ ràng gồm `Graduate` (1) và `Dropout` (0).
  * Biến can thiệp $T$: `Scholarship holder` (0/1).
  * Biến nền $Z$: Điểm tuyển sinh, nợ học phí, đóng học phí đúng hạn, học vấn phụ huynh, tuổi nhập học.

---

## 4. Hệ quả & Lợi ích (Consequences)
* **Tích cực:** Thỏa mãn hoàn toàn yêu cầu của Giảng viên về tính thực tế, quy mô đủ lớn để chạy chia 5 seed ghép cặp mà không bị rỗng mẫu; đảm bảo tính bao phủ (treatment overlap).
* **Tuân thủ quy định:** Đảm bảo điều kiện duyệt đề tài Midterm (Accept) và cung cấp cơ sở dữ liệu vững chắc cho bài báo khoa học Cuối kỳ.
