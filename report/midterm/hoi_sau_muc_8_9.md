# Câu hỏi sâu mục 8–10 (tách ra từ kịch bản)

### Cell 37 · markdown · 8.1 Thiết kế B0 / H1 / H2 + 8.2 Multi-treatment

**❓ Hỏi sâu:**
* *Vì sao không đổi cả hai thứ cùng lúc (mạng nơ-ron học cả sơ đồ lẫn bảng)?* — Nếu đổi hai thứ mà kết quả khác, không biết khác do cái nào. Thiết kế "đổi một thứ" giống thí nghiệm có đối chứng. Bản kết hợp có ở Table H3 (phụ lục, script riêng), không thuộc yêu cầu bắt buộc.

### Cell 41 · code · 8.A Config — mọi thiết lập ở một chỗ

**❓ Hỏi sâu:**
* *Siêu tham số chọn thế nào, có dò theo TEST không?* — Không. Hàm điểm của B0, $\lambda_1$ và $\tau$ của H2, ngưỡng F1 đều chọn trên VALIDATION. Phần còn lại cố định từ trước. TEST chỉ dùng một lần ở cuối.

### Cell 43 · code · 8.A Năm cách chia dữ liệu dùng chung

**❓ Hỏi sâu:**
* *545 hồ sơ TEST dùng vào hai việc khác nhau?* — Đúng. Nhãn tốt nghiệp của TEST ($Y_\text{test}$) chỉ dùng để chấm dự báo (Table A). Hồ sơ nền của TEST ($C_\text{test}$) dùng làm "quần thể chuẩn" để tính ATE (Table B) — không dùng nhãn.

### Cell 45 · code · 8.B B0 — học sơ đồ bằng HillClimb, chọn giám khảo trên VALIDATION

**❓ Hỏi sâu:**
* *So điểm BIC với điểm K2 trực tiếp không được sao?* — Hai loại điểm ở hai thang khác nhau, như điểm thang 10 với thang 100. Log-likelihood trên VAL là thước đo chung.
* *Ép học bổng → tốt nghiệp có phải áp đặt kết quả?* — Không. Cạnh bắt buộc chỉ tạo chỗ để đo; độ lớn do bảng xác suất học từ dữ liệu. Nếu học bổng vô dụng, hai cột có / không học bổng trong bảng sẽ bằng nhau và ATE ≈ 0. H2 không bị ép mà vẫn tự học ra cạnh này.
* *Vì sao chỉ cần cha của tốt nghiệp để tính ATE?* — Mặt nạ cấm học bổng → biến nền, nên mọi cha khác của tốt nghiệp đều là biến nền có trước học bổng → là một tập điều chỉnh cửa sau hợp lệ. Vì vậy $P(Y=1 \mid T=t, \text{cha khác})$ chính là $P(Y=1 \mid do(T=t), \text{cha khác})$.

### Cell 47 · code · 8.C H1 — thay bảng xác suất của tốt nghiệp bằng mạng nơ-ron

**❓ Hỏi sâu:**
* *Vì sao hội tụ sớm vậy, có học thiếu không?* — Đầu vào chỉ có $2^4 = 16$ tổ hợp, mạng thực chất chỉ cần học 16 xác suất; AUROC ngang B0 cho thấy nó học đủ.
* *Vì sao không cho mạng nhận tất cả biến?* — Slide 60 yêu cầu chỉ dùng cha của nút trong sơ đồ. Đưa biến khác vào sẽ phá cấu trúc nhân quả của mạng Bayes.

### Cell 49 · code · 8.D H2 — chọn $\lambda_1$, chọn ngưỡng $\tau$, kiểm tra trên seed 42

**❓ Hỏi sâu:**
* *Vì sao $\lambda_1$ chọn một lần ở seed 42, còn $\tau$ chọn từng seed?* — $\lambda_1$ là thiết lập huấn luyện, chọn một lần để tiết kiệm thời gian (mỗi lần học H2 khoảng 15–20 giây); $\tau$ chỉ là bước lọc sau khi học xong, rẻ, nên chọn lại cho từng seed.
* *Trọng số $W$ có phải là hiệu ứng nhân quả?* — Không (Slide 58). $W$ chỉ để quyết định có mũi tên; hiệu ứng tính lại bằng truy vấn can thiệp trên sơ đồ đã đóng băng.

### Cell 52 · code · 8.E Vòng lặp 5 lần chạy ghép cặp

**❓ Hỏi sâu:**
* *"Ghép cặp" nghĩa là gì, vì sao quan trọng?* — Trong mỗi seed ba mô hình dùng cùng TRAIN/VAL/TEST, nên hiệu giữa chúng loại được phần dao động do cách chia dữ liệu. Slide 39 yêu cầu ít nhất 5 lần chạy ghép cặp.
* *H1 seed 44 thấp hơn (0.218)?* — Đó là độ dao động của mạng nơ-ron giữa các lần chạy; Table C ghi độ tản H1 là 0.028, lớn nhất trong ba mô hình — mạng nơ-ron kém ổn định hơn bảng đếm.

### Cell 53 · code · 8.E2 Quá trình huấn luyện, overfit, có / không early stopping

**❓ Hỏi sâu:**
* *Sao loss TRAIN lại cao hơn VAL ở vài seed?* — Loss TRAIN đo trong lúc huấn luyện, khi dropout 0.2 đang bật nên bị đội lên; loss VAL đo ở chế độ đánh giá.
* *Sai số tái tạo tăng có phải H2 học kém đi?* — Không: phạt vòng tròn càng nặng thì mạng phải bỏ bớt mũi tên tạo vòng, chấp nhận khớp kém hơn chút để là một DAG hợp lệ.
* *B0 không có đường cong?* — B0 không học bằng gradient: tìm sơ đồ bằng leo đồi, bảng là đếm có làm mịn.

### Cell 57 · code · 8.F Refuter và chi phí — tạo Table C

**❓ Hỏi sâu:**
* *Sao "Original_Effect" của B0 trong Table C (0.231) khác Table B (0.242)?* — Table C dùng ước lượng **hồi quy tuyến tính của DoWhy** trên TRAIN với sơ đồ B0 (đúng tinh thần Slide 70–71: refuter kiểm tra sơ đồ + ước lượng hỗ trợ), còn Table B dùng bảng xác suất của mạng Bayes trên TEST. Hai cách khác nhau nên số hơi khác, nhưng cùng bậc.
* *Qua refuter là chứng minh mô hình đúng?* — Không. Refuter chỉ cho thấy kết quả không phải do may hay do vài điểm dữ liệu; không loại được nhiễu ẩn (phần đó ở mục 9.H).

### Cell 58 · code · Hiển thị Tables A, B, C, H3, Multi-Treatment

**❓ Hỏi sâu:**
* *CI hẹp vậy có nghĩa ATE rất chắc chắn?* — Không hẳn: CI giữ mô hình cố định. Bất định do học lại mô hình thể hiện ở độ tản giữa 5 seed (Table C).
* *Không cải thiện dự báo thì H1, H2 có ý nghĩa gì?* — Chúng kiểm chứng rằng kết luận nhân quả không phụ thuộc cách tham số hóa hay cách học sơ đồ. Kết quả âm tính đó chính là bằng chứng độ vững.

### Cell 59 · code · 8.2b Positivity và dấu hiệu nhân quả ngược của học phí

**❓ Hỏi sâu:**
* *Tương phản kép có cộng hưởng không?* — Không: ATE kết hợp (+0.67 đến +0.75) xấp xỉ tổng hai cái riêng.

### Cell 61 · code · Hình 8.1 — B0 và H2 học được sơ đồ gì

**❓ Hỏi sâu:**
* *Cột biến nền ở hình (c), (d) không theo thứ tự Z1 → Z5?* — Thứ tự xếp chỉ để mũi tên không đè nhau, không mang ý nghĩa thời gian.

### Cell 62 · code · Tính tay $q_m(t, C_i)$ để thấy công thức minh bạch

**❓ Hỏi sâu:**
* *Hồ sơ "có nợ + chưa đóng phí" có $q$ rất thấp (2–4%) dù có học bổng vẫn chỉ 16–37%?* — Nhóm này hiếm; đó là chỗ positivity yếu, con số dựa nhiều vào làm mịn.

### Cell 69 · code · 9.A Hai quần thể cho phần mở rộng

**❓ Hỏi sâu:**
* *Sao mục 9 dùng 7 hiệp biến mà mục 8 dùng 5?* — Mục 8 bị giới hạn bởi bảng xác suất rời rạc (thêm biến thì bảng phình to, thưa). Hồi quy ở mục 9 xử lý được nhiều biến hơn, nên thêm nơi ở và giới tính để kiểm tra kết luận không phụ thuộc bộ biến.

### Cell 71 · code · 9.B Kết cục 1 — điểm trung bình HK2

**❓ Hỏi sâu:**
* *Điểm HK2 có vấn đề gì không?* — Có một điểm cần nói thẳng: **22% sinh viên có điểm HK2 bằng 0**, và trong nhóm bỏ học là 51% — họ không còn học để có điểm. Vì vậy +1.48 một phần phản ánh việc *còn ở lại học*, không thuần là "học giỏi hơn".
* *Điểm HK2 xảy ra sau học bổng, dùng được không?* — Được, vì ở đây nó là **kết cục**, không phải biến điều chỉnh.

### Cell 73 · code · 9.C Kết cục 2 — tỷ lệ đạt tín chỉ HK2

**❓ Hỏi sâu:**
* *Sinh viên không đăng ký môn nào thì sao?* — 4% sinh viên có 0 tín chỉ đăng ký; mẫu số được đặt tối thiểu 1 nên tỷ lệ của họ là 0. Cùng lưu ý như điểm HK2: chỉ số này một phần phản ánh việc còn học.

### Cell 75 · code · 9.D Kết cục 3 — giữ chân toàn trường bằng AIPW

**❓ Hỏi sâu:**
* *Vì sao +16.9% nhỏ hơn +24%?* — Kết cục khác (ở lại gồm cả đang học, nên mức nền cao hơn) và quần thể khác. Điều quan trọng là vẫn dương rõ.
* *Vì sao chặn $e(x)$ ở 0.05–0.95?* — Người có $e$ gần 0 mà vẫn được học bổng sẽ nhận trọng số $1/e$ rất lớn và làm ước lượng nhảy lung tung; chặn để ổn định.

### Cell 77 · code · 9.E Trung gian — học bổng giúp qua đường nào

**❓ Hỏi sâu:**
* *Giả định gì?* — Thêm hai giả định: không có nhiễu ẩn giữa tín chỉ HK1 và tốt nghiệp (ví dụ động lực học tập ảnh hưởng cả hai), và quan hệ tuyến tính không có tương tác. Vì vậy 21.5% là ước lượng có điều kiện.
* *Gợi ý chính sách?* — Hỗ trợ học thuật sớm trong HK1 (khuyến nghị 2 ở mục 11).

### Cell 79 · code · 9.F CATE — học bổng giúp ai nhiều hơn

**❓ Hỏi sâu:**
* *Nhóm định trước nghĩa là gì?* — Chọn 6 nhóm trước khi nhìn kết quả, không dò nhóm nào có hiệu ứng lớn — tránh "câu cá" thống kê.

### Cell 83 · code · 9.H Robustness Value — kết luận có dễ bị lật không

**❓ Hỏi sâu:**
* *Bản cũ ghi 23.40%?* — Bản cũ dùng nhầm công thức xấp xỉ; đã sửa sang đúng công thức Cinelli–Hazlett.
* *Lưu ý ký hiệu:* ở mục này Z1 là tuổi, Z5 là học phí — khác cách đánh số của mục 8.

### Cell 86 · code · Table 10.1 phân bổ học bổng + Table 10.2 chuyển sang Kaggle

**❓ Hỏi sâu:**
* *+164 có đáng tin?* — Là ước lượng mô hình, không phải quan sát; cần cross-fitting và đánh giá off-policy (IPW/DR) mới chắc. Danh sách xếp hạng còn dùng biến học phí — biến có thể là hậu quả của bỏ học.
