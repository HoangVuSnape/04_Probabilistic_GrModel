# Nhận xét bảng và Insight từ Notebook

Nguồn: [00_MASTER_Causal_PGM_Educational_Intervention.ipynb](../00_MASTER_Causal_PGM_Educational_Intervention.ipynb)

## 1. Mục 1.1a — Lý do lựa chọn

**Vị trí trong notebook:** dòng JSON 143–155, anchor `s1-1a`.

Bảng giải thích lý do chọn dữ liệu UCI 697, treatment là học bổng, outcome là tốt nghiệp/bỏ học, năm biến nền `Z`, mô hình B0/H1/H2, năm run ghép cặp và các công cụ DoWhy/refuter/Cinelli–Hazlett.

**Nhận xét:** Thiết kế thực nghiệm cố gắng tách riêng ba yếu tố: cấu trúc đồ thị, cách tham số hóa và độ tin cậy của ước lượng nhân quả.

## 2. Mục 1.1b — Câu hỏi nghiên cứu

**Vị trí trong notebook:** dòng JSON 170–178, anchor `s1-1b`.

Bảng đặt ra sáu nhóm câu hỏi:

- `RQ0`: học bổng làm thay đổi xác suất tốt nghiệp bao nhiêu?
- `RQ1`: chênh lệch tốt nghiệp thô có bị phóng đại bởi nhiễu không?
- `RQ2`: H1 có cải thiện dự báo so với B0 không?
- `RQ3`: H2 có học được DAG hợp lệ không?
- `RQ4`: kết luận có vững trước nhiễu ẩn không?
- `RQ5–RQ6`: tác động qua kênh nào, nhóm nào hưởng lợi và phân bổ theo CATE có hiệu quả không?

**Nhận xét:** Notebook không chỉ đánh giá dự báo mà còn kiểm tra cấu trúc, estimand, trung gian, dị biệt tác động và chính sách phân bổ.

## 3. Mục 1.5 — Từ điển biến và vai trò nhân quả

**Vị trí trong notebook:** dòng JSON 505–514, anchor `s1-5`.

- `Y`: tốt nghiệp/bỏ học, đo cuối khóa.
- `T`: nhận học bổng, đo khi nhập học.
- `Z1–Z5`: các biến nền dùng để điều chỉnh.
- `M1`: số tín chỉ học kỳ 1, đóng vai trò trung gian và không được điều chỉnh trong mô hình chính.

**Nhận xét:** `Tuition fees up to date` có thời điểm đo không chắc chắn. Nếu đây là biến được cập nhật sau khi sinh viên có dấu hiệu bỏ học, điều chỉnh theo nó có thể gây bias. Tín chỉ học kỳ 1 phù hợp hơn với phân tích trung gian.

## 4. Mục 1.5 — Bảng giả định nhân quả

**Vị trí trong notebook:** dòng JSON 516–522, anchor `s1-5`.

Các giả định gồm:

- không còn nhiễu ẩn ngoài `Z`;
- `Z` được đo trước treatment;
- consistency;
- positivity/overlap;
- mỗi sinh viên chỉ xuất hiện một lần.

**Nhận xét:** Hai rủi ro lớn nhất là thời điểm đo của `Z3` và giả định không có nhiễu ẩn.

## 5. Mục 3.2 — So sánh thuật toán học cấu trúc

**Vị trí trong notebook:** dòng JSON 1423–1429, anchor `s3-2`.

Bảng so sánh PC, FCI, Hill Climb, GES và NOTEARS.

**Nhận xét:**

- Hill Climb phù hợp khi cần áp đặt ràng buộc miền mạnh.
- FCI xử lý nhiễu ẩn nhưng không tạo ra một DAG duy nhất.
- NOTEARS phù hợp với học cấu trúc bằng tối ưu liên tục và mặt nạ thời gian.
- Không thuật toán nào tự động giải quyết chiều nhân quả nếu thiếu ràng buộc miền.

Notebook cho thấy nếu chỉ cấm cạnh đi ra từ `Y` mà không cấm `T → Z`, thuật toán vẫn có thể học sai chiều thời gian.

## 6. Mục 4.2 — So sánh phương pháp học tham số

**Vị trí trong notebook:** dòng JSON 1630–1636, anchor `s4-2`.

Bảng so sánh MLE, BDeu, EM và Neural CPD.

**Nhận xét:**

- MLE dễ tạo xác suất bằng 0 trong các ô CPT thưa.
- BDeu phù hợp với mạng Bayes rời rạc nhờ làm mịn Dirichlet.
- EM phù hợp hơn khi dữ liệu bị thiếu.
- Neural CPD xử lý quan hệ phi tuyến và ô thưa tốt hơn, nhưng phức tạp hơn và không tự động làm kết luận nhân quả đáng tin cậy hơn.

## 7. Mục 11.1 — Tổng hợp câu trả lời nghiên cứu

**Vị trí trong notebook:** dòng JSON 8689–8696, anchor `s11-1`.

Kết quả chính:

- `RQ0`: ATE khoảng `+24` điểm phần trăm:
  - B0: `+0.2418`
  - H1: `+0.2374`
  - H2: `+0.2358`
- `RQ1`: chênh lệch thô `+34.54%`, lớn hơn ATE khoảng 10 điểm %, cho thấy confounding quan sát được đáng kể.
- `RQ2`: H1 gần như không cải thiện dự báo so với B0.
- `RQ3`: H2 học được DAG hợp lệ nhưng không cải thiện rõ rệt dự báo hay kết luận can thiệp.
- `RQ4`: Robustness Value khoảng `20.97%`.
- `RQ5`: khoảng `21.5%` hiệu ứng đi qua tín chỉ học kỳ 1; nhóm trên 20 tuổi có CATE cao hơn.
- `RQ6`: phân bổ theo CATE làm tăng số tốt nghiệp dự kiến trong mô phỏng, nhưng kết quả là in-sample.

# Insight chính

## Insight 1 — Hiệu ứng quan sát thô bị phóng đại

Chênh lệch tốt nghiệp quan sát được là `+34.54%`, nhưng ATE điều chỉnh chỉ khoảng `+24%`. Sinh viên nhận học bổng có thể có đặc điểm nền thuận lợi hơn; không thể dùng chênh lệch thô để kết luận toàn bộ mức tăng là do học bổng.

## Insight 2 — Mô hình phức tạp hơn chưa tạo thêm giá trị rõ ràng

H1 và H2 có ATE gần B0, trong khi mức cải thiện AUROC rất nhỏ. Với dữ liệu này, mô hình CBN minh bạch đã đủ để đưa ra kết luận chính; mô hình neural chưa chứng minh được lợi ích đáng kể.

## Insight 3 — Ràng buộc miền quan trọng hơn việc chỉ đổi thuật toán

Thiếu ràng buộc thời gian có thể tạo các cạnh phi lý như `Target_Graduate → Admission_grade` hoặc `Scholarship_holder → Debtor`. Chất lượng DAG phụ thuộc mạnh vào ràng buộc `Z → T → Y`.

## Insight 4 — Tín chỉ học kỳ 1 là một kênh trung gian

Tổng hiệu ứng khoảng `+21.61%`, gồm hiệu ứng trực tiếp `+16.96%` và hiệu ứng gián tiếp qua tín chỉ HK1 `+4.65%`. Khoảng 21.5% tổng hiệu ứng được giải thích qua kết quả học tập sớm.

## Insight 5 — Tác động không đồng nhất giữa các nhóm

Sinh viên trên 20 tuổi có CATE khoảng `+28.02%`, cao hơn nhóm không quá 20 tuổi khoảng 12 điểm phần trăm. Tuy nhiên, các CATE nhóm chưa có khoảng tin cậy nên chỉ nên xem là bằng chứng gợi ý.

## Insight 6 — Kết luận vẫn nhạy với nhiễu ẩn mạnh

Robustness Value khoảng `20.97%`, trong khi biến quan sát mạnh nhất có partial R² khoảng `10.41%`. Kết luận tương đối vững trước nhiễu yếu, nhưng vẫn có thể bị đảo ngược nếu tồn tại nhiễu ẩn mạnh.

## Insight 7 — Phân bổ theo CATE có tiềm năng nhưng chưa đủ để triển khai

Phân bổ theo CATE làm tăng số tốt nghiệp dự kiến khoảng 164 sinh viên trong mô phỏng. Tuy nhiên, kết quả dùng cùng dữ liệu để ước lượng và chấm điểm, chưa cross-fitting, chưa đánh giá off-policy và có sử dụng biến `Tuition fees up to date` đáng ngờ.

**Kết luận:** Nên xem kết quả phân bổ theo CATE là cơ sở để thiết kế thử nghiệm chính sách có kiểm soát, không phải bằng chứng đủ để triển khai đại trà.
