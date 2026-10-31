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

### 3.1. Cập nhật từ data dictionary UCI

Trang UCI chính thức xác nhận dữ liệu có 36 thuộc tính, không có missing value,
bao gồm thông tin hồ sơ nhập học, nhân khẩu học, kinh tế-xã hội và kết quả học
tập theo học kỳ. Vì mục tiêu là ước lượng tác động của học bổng tại thời điểm
đầu kỳ, các biến `Curricular units 1st sem ...` và `Curricular units 2nd sem ...`
được đánh dấu là hậu treatment; không đưa chúng vào tập điều chỉnh admission-time.

Từ ngày 2026-10-30, quyết định DAG được chia thành hai tầng:

1. **DAG chính để thực thi B0/H1/H2 (primary estimand):** giữ tập $Z$ năm biến
   đã chốt ở trên để tránh sparse cells và bảo đảm khả năng giải thích.
2. **DAG mở rộng để sensitivity analysis:** xem xét thêm `Application mode`,
   `Application order`, `Course`, `Previous qualification`,
   `Previous qualification (grade)`, `Father's qualification`,
   `Mother's occupation`, `Father's occupation`, `Gender`, `Displaced`,
   `Educational special needs` và `International`. Chỉ đưa biến nào qua được
   kiểm tra temporal ordering, overlap và leakage vào sensitivity run.

Các cạnh nhân quả nền tảng được giả định theo thứ tự:

```text
Z_baseline -> Scholarship holder -> Target
Z_baseline --------------------> Target
```

Không cho phép `Target -> Scholarship holder`, `Target -> Z_baseline` hoặc
`Scholarship holder -> Z_baseline`. Đây là giả định domain DAG để định danh,
không phải kết luận được chứng minh chỉ từ tương quan dữ liệu.

---

## 4. Hệ quả & Lợi ích (Consequences)
* **Tích cực:** Thỏa mãn hoàn toàn yêu cầu của Giảng viên về tính thực tế, quy mô đủ lớn để chạy chia 5 seed ghép cặp mà không bị rỗng mẫu; đảm bảo tính bao phủ (treatment overlap).
* **Tuân thủ quy định:** Đảm bảo điều kiện duyệt đề tài Midterm (Accept) và cung cấp cơ sở dữ liệu vững chắc cho bài báo khoa học Cuối kỳ.
* **Quyết định thực thi:** Chạy primary estimand với DAG tối giản trước; chỉ chạy
  DAG mở rộng sau khi primary pipeline pass toàn bộ validation.
