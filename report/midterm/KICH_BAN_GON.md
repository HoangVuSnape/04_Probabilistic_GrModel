# Kịch bản nói — bản gọn

**21 điểm dừng** thay cho 89 cell. Mỗi điểm dừng gom các cell liên quan; cell chỉ chạy kỹ thuật (nạp thư viện, cấu hình, xuất bảng) được lướt qua bằng một câu.
Mỗi điểm dừng có **🎤 Nói** (đọc gần như nguyên văn) và **👉 Chỉ vào** (thứ cần trỏ trên màn hình). Từ khóa khó: xem tab **Glossary** hoặc **Công thức**. Bản đầy đủ từng cell: tab **Kịch bản đầy đủ**.

| Phần | Điểm dừng | Thời lượng gợi ý |
| :--- | :--- | :--- |
| Mục 1 — Dữ liệu và câu hỏi | ① – ④ | 4 phút |
| Mục 2–6 — Nền tảng | ⑤ – ⑧ | 4 phút |
| Mục 7 — DoWhy | ⑨ | 1 phút |
| **Mục 8 — B0 / H1 / H2 (phần chính)** | ⑩ – ⑮ | 7 phút |
| Mục 9 — Mở rộng | ⑯ – ⑱ | 3 phút |
| Mục 10–11 — Chính sách và kết luận | ⑲ – ㉑ | 3 phút |

---

## Mục 1 — Dữ liệu và câu hỏi

### ① Cell 0–3 · Mở đầu, lý do chọn, câu hỏi nghiên cứu

**🎤 Nói:**
"Em xin trình bày đồ án môn Mô hình đồ thị xác suất. Câu hỏi rất đời thường: *nếu nhà trường cấp học bổng cho một sinh viên, khả năng tốt nghiệp tăng thêm bao nhiêu?* Dữ liệu là bộ thật UCI 697. Toàn bộ đồ án nằm trong một notebook, bấm Restart & Run All là sinh lại mọi bảng và hình.

Em có dùng AI như một lập trình viên đi cùng; các quyết định phương pháp — thứ tự thời gian của biến, cạnh bị cấm, tham số làm mịn, nguyên tắc không đụng nhãn TEST — do em quyết định và chịu trách nhiệm.

Em chọn học bổng làm can thiệp vì đây là thứ nhà trường *thực sự bật/tắt được*. Kết cục là tốt nghiệp so với bỏ học. Năm biến nền đo lúc nhập học, ảnh hưởng cả việc nhận học bổng lẫn tốt nghiệp. Câu hỏi chính được viết thành ATE; sáu câu phụ ghi ở bảng, và mục 11 sẽ trả lời từng câu."

**👉 Chỉ vào:** mục lục (bấm được), bảng lý do lựa chọn 1.1a, bảng câu hỏi RQ0–RQ6 ở 1.1b.

### ② Cell 4–9 · Dữ liệu, từ điển biến, luồng loại trừ

**🎤 Nói:**
"Bộ gốc có 4,424 sinh viên. Em loại 794 sinh viên còn đang học vì họ chưa có kết cục — gán đỗ hay trượt đều là đoán. Còn lại **3,630** người: 60.85% tốt nghiệp, 26.69% có học bổng. Không có dòng trùng, không có giá trị thiếu.

Bảng từ điển biến là *hợp đồng* của đồ án: mỗi biến có cột gốc, cách mã hóa, thời điểm đo và vai trò. Em đánh dấu thẳng điểm yếu: biến *học phí đúng hạn* có thời điểm đo mơ hồ, nên chỉ là giả định."

**👉 Chỉ vào:** output 3,630 / 60.85%; bảng từ điển biến (dòng học phí in đậm "mơ hồ"); bảng luồng loại trừ.

**⚠️** Nếu hỏi loại Enrolled có gây chệch không: mục 9.D kiểm tra lại trên cả 4,424 SV, kết quả vẫn dương (+16.87%).

### ③ Cell 10–15 · Kiểm toán trước khi mô hình hóa

**🎤 Nói:**
"Bốn kiểm tra trước khi dựng mô hình.
Một, 12 biến tín chỉ và điểm HK1, HK2 xảy ra *sau* học bổng nên bị loại khỏi tập điều chỉnh.
Hai, ngưỡng chia điểm vào 120 / 140 cho ba nhóm đủ mẫu; nhóm dưới 120 tốt nghiệp thấp hẳn (52.6% so với khoảng 64%) và tỷ lệ học bổng cũng khác nhau — điểm vào ảnh hưởng cả hai phía, đúng là biến nhiễu.
Ba, học phí đúng hạn liên kết với tốt nghiệp mạnh nhất, mạnh hơn cả học bổng — đây là tín hiệu cảnh báo chứ không phải lý do chọn biến.
Bốn, cả 6 tầng điểm vào × tuổi đều có người có và không có học bổng."

**👉 Chỉ vào:** Hình 1.1 (hai vạch đỏ 120 / 140); Hình 1.2 (thanh đỏ học phí đứng đầu); Bảng 1.6b (cột số người có học bổng).

### ④ Cell 16–19 · Nghịch lý quan sát thô

**🎤 Nói:**
"Nhìn thô: có học bổng 86.17% tốt nghiệp, không có 51.63%, chênh **34.54 điểm %**. Nhưng không thể kết luận cấp học bổng tăng 35 điểm, vì người có học bổng vốn đã khác — điểm cao hơn, ít nợ hơn. Đây là $P(Y \mid T)$, còn câu hỏi của em là $P(Y \mid do(T))$. Phần sau sẽ trừ đi sự khác biệt sẵn có đó."

**👉 Chỉ vào:** Hình 2.1 trái (86.2% vs 51.6%); bên phải ô đỏ n = 37 (sẽ quay lại ở mục 8).

---

## Mục 2–6 — Nền tảng

### ⑤ Cell 20–23 · Học cấu trúc cần ràng buộc thời gian

**🎤 Nói:**
"Theo Slide 57, B0 phải *học* đồ thị bằng thuật toán; em chọn HillClimb vì dữ liệu rời rạc và nó nhận ràng buộc dễ nhất. Hình 3.1 cho thấy vì sao cần ràng buộc: học tự do ra cạnh vô lý *tốt nghiệp → điểm đầu vào*. Cấm cạnh đi ra từ Y thì hết, nhưng còn *học bổng → Debtor* vì chưa cấm học bổng → biến nền. Bài học: phải khai báo đủ thứ tự biến nền → học bổng → tốt nghiệp, và đó là cách B0 ở mục 8 làm."

**👉 Chỉ vào:** Hình 3.1, các cạnh đỏ ở (a) và (b).

### ⑥ Cell 24–26 · Học tham số: MLE và BDeu

**🎤 Nói:**
"Mỗi nút có một bảng xác suất, gọi là CPD. MLE chỉ đếm thô nên ô ít dữ liệu dễ ra 0% hay 100%. BDeu cộng thêm vài 'mẫu ảo' để làm mịn; mục 8 dùng ESS = 5, rất nhỏ so với 2,541 mẫu TRAIN nên chỉ tác động lên ô hiếm."

**👉 Chỉ vào:** hai bảng CPD in ra (0.597 so với 0.594 — ô này đủ đông nên gần như không đổi).

### ⑦ Cell 27–30 · Bốn kiểm tra hợp lệ của DAG

**🎤 Nói:**
"Trước khi suy luận, đồ thị phải qua bốn kiểm tra: không có chu trình, không có cạnh ngược thời gian, đúng danh sách cạnh cấm và bắt buộc, và tập biến nền chặn được mọi đường cửa sau. Domain DAG của em đạt cả bốn."

**👉 Chỉ vào:** bốn dòng PASSED.

### ⑧ Cell 31–33 · Domain DAG và phẫu thuật đồ thị

**🎤 Nói:**
"Đây là đồ thị em vẽ theo hiểu biết, dùng cho DoWhy ở mục 7 và làm tham chiếu. Bên trái: các đường cửa sau từ biến nền đi vào học bổng là nguồn của 34.54 điểm thô. Bên phải: phép $do(T)$ cắt mọi mũi tên đi vào học bổng; điều chỉnh theo biến nền cho phép tái tạo đúng tình huống đó từ dữ liệu quan sát. Tín chỉ HK1 là trung gian nên không điều chỉnh."

**👉 Chỉ vào:** Hình 6.1, mũi tên đỏ nét đứt bị cắt ở (b).

**⚠️** Đồ thị này **không** dùng để chạy B0/H1/H2 ở Tables A–C; B0 tự học đồ thị (mục 8).

---

## Mục 7 — DoWhy

### ⑨ Cell 34–36 · Đo thử bằng DoWhy và ba phép thử phá

**🎤 Nói:**
"Đưa Domain DAG vào DoWhy: nó tự tìm đúng 5 biến nền cần điều chỉnh. Hiệu ứng: **+24.12%** bằng hồi quy, +25.16% bằng trọng số xu hướng. Ba phép thử phá đều qua: thêm biến nhiễu ngẫu nhiên vẫn +0.241; thay học bổng bằng biến giả thì về gần 0 (+0.005); dùng 80% dữ liệu vẫn +0.245. Qua refuter chỉ chứng minh ước lượng ổn định, không chứng minh không có nhiễu ẩn — phần đó ở mục 9."

**👉 Chỉ vào:** dòng +24.12% và ba dòng VƯỢT QUA.

---

## Mục 8 — Ba mô hình B0, H1, H2 (phần chính)

### ⑩ Cell 37 · Thiết kế: mỗi mô hình chỉ đổi đúng một thứ

**🎤 Nói:**
"Mạng Bayes gồm sơ đồ mũi tên và bảng xác suất.
**B0**: HillClimb học sơ đồ, BDeu điền bảng.
**H1**: giữ sơ đồ của B0, chỉ thay bảng của tốt nghiệp bằng mạng nơ-ron.
**H2**: mạng nơ-ron NOTEARS học sơ đồ, bảng vẫn điền bằng BDeu như B0.
Cả ba dùng cùng 7 biến, cùng dữ liệu, cùng luật cấm trong mỗi lần chạy, nên mọi khác biệt là do đúng một thay đổi. Mục 8.2 là phần khám phá thêm hai can thiệp khác; em giới thiệu ở đây và trình bày kết quả sau Tables A–C."

**👉 Chỉ vào:** ba gạch đầu dòng B0 / H1 / H2 ở mục 8.1.

*(Cell 38–43: nạp thư viện, cấu hình, mã hóa 7 biến, 5 split 70/15/15 phân tầng — lướt qua: "mọi thiết lập nằm trong một Config, 5 split được lưu ra file và đóng băng".)*

### ⑪ Cell 44–49 · Ba mô hình trên seed 42

**🎤 Nói:**
"**B0**: thử ba hàm điểm BIC, BDeu, K2 rồi để tập VALIDATION quyết định. K2 thắng ở cả 5 seed vì nó giữ được cạnh Debtor → tốt nghiệp mà hai hàm kia bỏ, và sơ đồ có cạnh đó đoán dữ liệu mới tốt hơn. Cha của tốt nghiệp: học bổng, Debtor, học phí, tuổi.
**H1**: mạng 769 tham số chỉ nhận 4 cha đó; dừng sớm ở epoch 3 vì đầu vào chỉ có 16 tổ hợp.
**H2**: mặt nạ chặn sẵn các cạnh cấm; ràng buộc vòng tròn về $3.3 \times 10^{-7}$; ngưỡng cạnh chọn trên VALIDATION. Ở ngưỡng được chọn, cạnh học bổng → tốt nghiệp được **tự học ra**, không phải ép.
ATE seed 42: B0 +0.247, H1 +0.245, H2 +0.243."

**👉 Chỉ vào:** bảng 3 ứng viên B0 (K2 −2208 tốt nhất); dòng "best epoch" của H1; bảng chọn τ của H2 (cột T->Y_forced = False).

### ⑫ Cell 50–55 · Năm lần chạy ghép cặp và quá trình huấn luyện

**🎤 Nói:**
"Lặp lại cho 5 seed, cùng split cho cả ba mô hình. ATE mọi lần đều trong khoảng 0.22 đến 0.25.
Quá trình huấn luyện ổn: loss VALIDATION của H1 chạm đáy rất sớm và early stopping cắt đúng chỗ; H2 hội tụ ở cả 5 seed. AUROC trên TRAIN, VAL, TEST gần như bằng nhau nên không overfit. Bỏ early stopping thì loss VAL và TEST tăng nhẹ — đúng dấu hiệu bắt đầu overfit."

**👉 Chỉ vào:** dòng tóm tắt từng seed; Hình 8.2 (a) đường loss; Bảng 8.E2 cột chênh TRAIN − TEST.

### ⑬ Cell 56–58 + 65 · Tables A, B, C — kết quả chính

**🎤 Nói:**
"**Table A — dự báo**: AUROC B0 0.800, H1 0.801, H2 0.803; hiệu ghép cặp nhỏ hơn độ dao động giữa các lần chạy, nên mạng nơ-ron **không cải thiện rõ** dự báo — đây là null finding hợp lệ.
**Table B — can thiệp**: ATE B0 +0.242, H1 +0.237, H2 +0.236. **Kết luận không đổi giữa ba mô hình.** So với 34.5 điểm thô, khoảng 10 điểm là thiên lệch do khác biệt sẵn có.
**Table C — độ vững**: thay học bổng bằng biến giả thì hiệu ứng về khoảng 0; dùng 80% dữ liệu vẫn khoảng 0.23; B0 chạy chưa tới nửa giây, H2 đắt nhất."

**👉 Chỉ vào:** dòng Mean ± Std của Table A (AUROC) và Table B (ATE); cột Placebo, Subset của Table C.

### ⑭ Cell 59–60 · Can thiệp khác: học phí đúng hạn (khám phá)

**🎤 Nói:**
"Mục 8.2 thử thêm can thiệp *học phí đúng hạn*. Con số +45 đến +51% là quá lớn để tin. Bảng 4 ô cho thấy lý do: người không học bổng và chưa đóng đúng hạn chỉ 4.5% tốt nghiệp — giống 'bỏ học rồi ngừng đóng phí' hơn là ngược lại. Còn ô vừa có học bổng vừa chưa đóng đúng hạn chỉ 37 người, điều kiện dương yếu. Vì vậy em chỉ kết luận về học bổng; bốn bộ ước lượng cổ điển cho học bổng khoảng +21 đến +23%, khớp với mục 8."

**👉 Chỉ vào:** bảng 4 ô (4.5%, n = 37).

### ⑮ Cell 61–64 · B0 và H2 học được gì, và forest plot

**🎤 Nói:**
"Hình 8.1: B0 học thêm chuỗi cạnh giữa các biến nền; H2 thưa hơn, gần như chỉ giữ các cạnh vào học bổng và tốt nghiệp. Trọng số cạnh không phải hiệu ứng nhân quả; hiệu ứng được tính lại bằng truy vấn can thiệp, và cell 62 tính tay lại cho vài hồ sơ để thấy công thức minh bạch, không dùng nhãn TEST. Hình 8.3: mọi seed của cả ba mô hình đều quanh +0.24, cách xa vạch chênh lệch thô."

**👉 Chỉ vào:** Hình 8.1 (c) và (d); bảng $q_m(t, C_i)$; Hình 8.3 vạch xám thô.

---

## Mục 9 — Phân tích mở rộng

### ⑯ Cell 66–75 · Học bổng giúp ở những mặt nào

**🎤 Nói:**
"Đổi sang các kết cục khác, kết luận vẫn cùng hướng: điểm HK2 tăng **1.48** trên thang 20 (thô là 2.63, điều chỉnh làm giảm gần một nửa); tỷ lệ đạt tín chỉ tăng **14.5 điểm**; tỷ lệ ở lại trường tính trên cả 4,424 sinh viên, kể cả người đang học, tăng **16.87 điểm** bằng AIPW. Kết quả cuối này cũng trả lời câu hỏi việc loại Enrolled có làm chệch không."

**👉 Chỉ vào:** ba dòng ATE in ra ở 9.B, 9.C, 9.D.

### ⑰ Cell 76–81 · Học bổng giúp bằng đường nào, và giúp ai nhiều hơn

**🎤 Nói:**
"Khoảng **21.5%** tác dụng đi qua việc học tốt hơn ở học kỳ 1; phần còn lại là trực tiếp. Theo nhóm: sinh viên **trên 20 tuổi** được lợi nhiều nhất, +28 điểm so với +16 ở nhóm đúng tuổi. Với nhóm có nợ, hiệu ứng tuyệt đối không lớn hơn; cột '2.03x' chỉ là tỷ số rủi ro vì mức nền thấp. Các nhóm chưa có khoảng tin cậy nên em đọc thận trọng."

**👉 Chỉ vào:** dòng tỷ lệ trung gian 21.5%; bảng CATE (hai dòng tuổi).

### ⑱ Cell 82–84 · Kết luận có dễ bị lật không

**🎤 Nói:**
"Robustness Value bằng **20.97%**: một biến ẩn phải giải thích ít nhất khoảng 21% phần dư của *cả* việc nhận học bổng lẫn tốt nghiệp mới xóa được hiệu ứng. Biến quan sát mạnh nhất là học phí chỉ khoảng 10%. Vậy kết luận vững trước nhiễu ẩn yếu hơn học phí, nhưng không miễn nhiễm với nhiễu mạnh gấp đôi."

**👉 Chỉ vào:** dòng $RV_{q=1}$ = 20.97% và bảng mốc partial $R^2$ (học phí 10.41%).

---

## Mục 10–11 — Chính sách và kết luận

### ⑲ Cell 85–87 · Phân bổ học bổng và chuyển sang dữ liệu khác

**🎤 Nói:**
"Giữ nguyên 969 suất nhưng ưu tiên người được dự đoán hưởng lợi nhiều nhất, tỷ lệ tốt nghiệp dự kiến tăng từ 60.85% lên 65.37%, tức khoảng **+164 sinh viên**; ưu tiên theo nợ chỉ +49, ngẫu nhiên +37. Con số này lạc quan vì cùng mô hình vừa chọn vừa chấm, chưa cross-fit. Thử trên Kaggle cho kết quả trái chiều, và Kaggle là dữ liệu tổng hợp, nên chưa chứng minh được kết quả đúng ở trường khác."

**👉 Chỉ vào:** Table 10.1 (dòng CATE Optimal +164); Table 10.2.

### ⑳ Cell 88 · Trả lời câu hỏi nghiên cứu

**🎤 Nói:**
"Trả lời ngắn: học bổng làm tăng khả năng tốt nghiệp khoảng **24 điểm %**, không phải 34.5; kết quả giống nhau ở B0, H1, H2; mạng nơ-ron không cải thiện dự báo rõ rệt; kết luận vượt qua các phép thử và khá vững trước nhiễu ẩn. Hạn chế lớn nhất em tự nhận là biến học phí đúng hạn, có thể là hậu quả của bỏ học."

**👉 Chỉ vào:** bảng trả lời RQ0–RQ6 ở mục 11.1.

### ㉑ Cell 88 · Khuyến nghị và kết thúc

**🎤 Nói:**
"Bốn khuyến nghị: thí điểm phân bổ học bổng theo mức hưởng lợi; hỗ trợ học thuật sớm trong học kỳ 1 vì khoảng 21.5% tác động đi qua kỳ này; dùng các phép thử phá làm quy chuẩn kiểm định chính sách; và dùng mạng Bayes B0 làm lõi cho hệ thống cảnh báo sớm vì nó dự báo ngang mạng nơ-ron mà minh bạch hơn. Em xin cảm ơn thầy và sẵn sàng nhận câu hỏi."

**👉 Chỉ vào:** mục 11.2.

**⚠️** Nếu hỏi "vậy làm H1/H2 để làm gì?": để kiểm chứng kết luận không phụ thuộc cách tham số hóa hay cách học đồ thị — kết quả âm tính ấy chính là bằng chứng độ vững.
