# Mục 9: mục tiêu, input, output và ý nghĩa từng phần

**Mục tiêu chung:** học bổng có làm được gì ngoài tỷ lệ tốt nghiệp? Mục này đo thêm kết cục khác, tìm đường đi của tác dụng, xem ai được lợi nhiều hơn, và kiểm tra độ vững.

## 9.A: Chuẩn bị hai quần thể (Cell 69)

- **Mục tiêu:** tạo đúng bộ dữ liệu cho từng phép đo.
- **Input:** `data/uci_697.csv`, 4,424 SV. Biến cần dùng: học bổng, tốt nghiệp, điểm HK2, tín chỉ HK2, tín chỉ HK1, và 7 biến điều chỉnh (tuổi, điểm vào, mẹ có bằng, Debtor, học phí đúng hạn, displaced, giới tính).
- **Output:**
    - `df_completed`: 3,630 SV có kết cục, dùng cho 9.B, 9.C, 9.E, 9.F.
    - `df_full`: 4,424 SV, dùng cho 9.D.
- **Ý nghĩa:** giữ người đang học ở bước giữ chân để kiểm tra việc loại họ ở mục 8 có làm lệch kết luận không.

## 9.B: Điểm HK2 (Cell 71)

- **Mục tiêu:** học bổng có làm điểm học kỳ 2 tăng không?
- **Input:** `df_completed`; kết cục là điểm HK2 (thang 20); học bổng và 7 biến điều chỉnh.
- **Làm gì:** hồi quy OLS: điểm ~ học bổng + 7 biến. Hệ số của học bổng là ATE.
- **Output:**
    - ATE = **+1.478 điểm**, CI 95% [1.093, 1.863], p ≈ 7×10⁻¹⁴.
    - Chênh lệch thô là +2.63, nên điều chỉnh làm giảm gần một nửa.
- **Ý nghĩa:** học bổng có tác dụng tích cực lên học tập. Tuy nhiên 22% SV có điểm bằng 0, và trong nhóm bỏ học tỷ lệ này là 51%, nên một phần kết quả phản ánh việc còn đi học.

## 9.C: Tỷ lệ đạt tín chỉ HK2 (Cell 73)

- **Mục tiêu:** học bổng có giúp sinh viên đạt nhiều môn đăng ký hơn không?
- **Input:** `df_completed`; kết cục là tín chỉ đạt chia tín chỉ đăng ký (mẫu số tối thiểu 1); học bổng và 7 biến.
- **Làm gì:** hồi quy OLS như 9.B.
- **Output:**
    - ATE = **+14.50 điểm %**, CI [11.80%, 17.20%].
    - Tỷ lệ đạt 100% số môn đăng ký: 33.9% (không học bổng) so với 58.2% (có học bổng), số thô chưa điều chỉnh.
- **Ý nghĩa:** học bổng giảm việc nợ tín chỉ. Đây cũng là kết cục sau can thiệp nên không được dùng làm biến điều chỉnh.

## 9.D: Giữ chân toàn trường bằng AIPW (Cell 75)

- **Mục tiêu:** học bổng có làm sinh viên ở lại trường hơn không, tính trên toàn bộ 4,424 SV?
- **Input:** `df_full`; kết cục giữ chân = 1 nếu không bỏ học (kể cả người đang học); học bổng và 7 biến.
- **Làm gì:**
    1. Mô hình xác suất được học bổng (logistic), giới hạn trong [0.05, 0.95].
    2. Hai mô hình kết cục (logistic) cho nhóm có và không có học bổng.
    3. Tính điểm AIPW cho từng SV: dự đoán cộng với phần sửa sai có trọng số.
    4. Bootstrap 1,000 lần để có khoảng tin cậy.
- **Output:**
    - Ở lại trường: 64.28% (không học bổng) so với 81.15% (có học bổng).
    - ATE = **+16.87%**, CI [13.79%, 19.73%].
- **Ý nghĩa:** kết quả dương rõ, và không cần loại người đang học. AIPW không chệch nếu một trong hai mô hình đúng.

## 9.E: Trung gian: học bổng đi qua đường nào? (Cell 77)

- **Mục tiêu:** bao nhiêu tác dụng của học bổng đi qua việc học tốt hơn ở học kỳ 1?
- **Input:** `df_completed`; trung gian M = tín chỉ HK1 đạt; kết cục Y = tốt nghiệp; học bổng và 7 biến.
- **Làm gì:**
    1. Hồi quy M ~ T + biến: hệ số a = **+0.729 tín chỉ** (học bổng làm tăng tín chỉ).
    2. Hồi quy Y ~ T + M + biến: hệ số của T là NDE (tác dụng trực tiếp), hệ số của M là b.
    3. NIE = a × b.
- **Output:**
    - Tổng hiệu ứng = **+21.61%**.
    - NDE = +16.96% (78.5%).
    - NIE = +4.65% (**21.5%**).
- **Ý nghĩa:** khoảng một phần năm tác dụng đi qua học kỳ 1. Điều này dẫn tới khuyến nghị hỗ trợ học thuật sớm.
- **Giả định:** không có nhiễu ẩn giữa M và Y, và không có tương tác.

## 9.F: Ai được lợi nhiều hơn? (CATE, Cell 79)

- **Mục tiêu:** hiệu ứng khác nhau ra sao giữa các nhóm?
- **Input:** `df_completed`; 6 nhóm định trước: tuổi (trên và dưới 20), Debtor (có và không), nơi ở (gia đình và thuê trọ xa).
- **Làm gì:** trong mỗi nhóm, khớp logistic tốt nghiệp ~ học bổng + biến. Đặt T=0 cho mọi người rồi tính tỷ lệ dự đoán; làm tương tự với T=1; hiệu là CATE.
- **Output (tóm tắt):**

|Nhóm|Không học bổng|Có học bổng|CATE|
|---|---|---|---|
|Trên 20 tuổi|38.7%|66.7%|**+28.0**|
|Đúng tuổi|68.6%|84.6%|+16.1|
|Có nợ|19.7%|40.0%|+20.3|
|Không nợ|60.0%|81.4%|+21.4|
|Ở gia đình|48.3%|74.2%|+25.9|
|Thuê trọ xa|61.4%|79.0%|+17.6|

- **Ý nghĩa:** nhóm trên 20 tuổi được lợi nhiều nhất. Cột "tỷ số" (ví dụ 2.03x) là tỷ số rủi ro, không phải "lợi gấp n lần". Chưa có CI cho từng nhóm nên đọc thận trọng.

## 9.G: Tổng hợp (Cell 81)

- **Mục tiêu:** gom kết quả 9.B đến 9.E vào một bảng.
- **Input:** kết quả của các phần trước.
- **Output:** `Table_Alternative_Outcomes.csv` và `Table_Heterogeneous_CATE.csv`.
- **Ý nghĩa:** bốn chiều kết quả đều dương, nên kết luận "học bổng có lợi" không phụ thuộc vào kết cục nào được chọn. Lưu ý: cột p-value trong bảng này ghi chữ cố định như "< 1e-10", không phải giá trị tính lại.

## 9.H: Độ nhạy trước nhiễu ẩn (Cell 83)

- **Mục tiêu:** một biến chưa đo được phải mạnh cỡ nào để lật kết luận?
- **Input:** `df_uci` (3,630 SV); hồi quy Y ~ T + 5 biến nền, mã hóa riêng cho mục này (Z1 tuổi, Z2 điểm vào trên trung vị, Z3 mẹ trên mức cơ bản, Z4 Debtor, Z5 học phí).
- **Làm gì:**
    1. Lấy hệ số và thống kê t của học bổng: β = +0.2268, t = 14.20, df = 3,623.
    2. Tính f = |t| / √df, rồi RV = 0.5 × (√(f⁴ + 4f²) − f²).
    3. Tính thêm RV ở mức ý nghĩa 5%.
    4. Tính partial R² của từng biến để làm mốc so sánh.
- **Output:**
    - **RV = 20.97%**, RV (α = 0.05) = 18.37%.
    - Partial R² của học bổng 5.27%.
    - Mốc: học phí đúng hạn 10.41%, tuổi 4.22%, Debtor 0.81%, điểm vào 0.48%, mẹ 0.03%.
- **Ý nghĩa:** một biến ẩn phải giải thích ít nhất 21% phần dư của cả học bổng và tốt nghiệp mới xóa được hiệu ứng. Biến đo được mạnh nhất (học phí) chỉ ở mức 10%, nên kết luận khá vững, nhưng không miễn nhiễm với nhiễu mạnh gấp đôi học phí.

## Tóm tắt một dòng cho mỗi phần

| Phần | Hỏi gì                        | Kết quả chính       |
| ---- | ----------------------------- | ------------------- |
| 9.A  | Dữ liệu nào cho từng câu hỏi? | 3,630 và 4,424 SV   |
| 9.B  | Điểm HK2 thế nào?             | +1.48 điểm          |
| 9.C  | Tín chỉ thế nào?              | +14.5 điểm %        |
| 9.D  | Ở lại trường thế nào?         | +16.9 điểm %        |
| 9.E  | Đi qua đường nào?             | 21.5% qua học kỳ 1  |
| 9.F  | Ai được lợi?                  | Trên 20 tuổi (+28)  |
| 9.G  | Tổng hợp                      | Bốn chiều đều dương |
| 9.H  | Có vững không?                | RV = 20.97%         |
