# Glossary "ăn liền" — hiểu nhanh các từ khóa trong notebook

Bản đọc nhanh, không cần nền toán. Mỗi từ có **một câu giải thích**, **một ví dụ đời thường hoặc ví dụ trong dự án**, và **câu nói để trả lời thầy** khi bị hỏi.
Muốn xem công thức đầy đủ: [glossary.md](glossary.md).

> **Câu chuyện của cả đồ án trong 3 dòng**
> Sinh viên có học bổng tốt nghiệp nhiều hơn hẳn (86% so với 52%, chênh **34.5 điểm %**).
> Nhưng người có học bổng vốn đã "khác" (điểm vào cao hơn, ít nợ hơn…), nên không thể nói học bổng gây ra toàn bộ 34.5 điểm đó.
> Sau khi dùng đồ thị nhân quả để "trừ" phần khác biệt đó ra, hiệu ứng thật của học bổng còn khoảng **+24 điểm %** — và kết quả này giống nhau ở cả 3 mô hình B0, H1, H2.

---

## Phần 1 — Ba nhân vật chính

| Từ khóa | Hiểu nhanh | Trong dự án |
| :--- | :--- | :--- |
| **T (treatment / can thiệp)** | Thứ mình muốn biết "nếu làm thì có tác dụng không". | Có học bổng hay không. |
| **Y (outcome / kết cục)** | Kết quả mình quan tâm. | Tốt nghiệp (1) hay bỏ học (0). |
| **Z (confounder / biến gây nhiễu)** | Yếu tố ảnh hưởng **cả** việc nhận học bổng **lẫn** việc tốt nghiệp, làm mình dễ nhầm tương quan thành nhân quả. | Điểm đầu vào, đang nợ học phí, đóng học phí đúng hạn, học vấn của mẹ, tuổi > 20. |

**Ví dụ đời thường cho Z:** người mang ô thường đi đường ướt. Ô không làm đường ướt — **trời mưa** (Z) gây ra cả hai.

**Nói với thầy:** *"Z là các biến đo lúc nhập học, trước khi có học bổng, ảnh hưởng cả T lẫn Y; phải điều chỉnh theo chúng thì mới tách được hiệu ứng thật."*

---

## Phần 2 — Đồ thị nhân quả

**DAG (đồ thị có hướng không chu trình)**
Bản vẽ "cái gì gây ra cái gì" bằng mũi tên, và không có vòng tròn (A gây B, B lại gây A).
*Trong dự án:* Hình 6.1 — các mũi tên Z → T, Z → Y, T → Y.
*Nói với thầy:* "DAG là giả định của nhóm về cơ chế, không phải thứ dữ liệu tự chứng minh."

**Domain DAG (đồ thị chuyên gia)**
DAG do con người vẽ dựa trên hiểu biết và thứ tự thời gian, trước khi nhìn dữ liệu.
*Trong dự án:* 7 nút, 11 cạnh; dùng làm **tham chiếu**, còn B0 thì tự học DAG từ dữ liệu.

**Trật tự thời gian**
Chuyện xảy ra sau không thể gây ra chuyện xảy ra trước.
*Ví dụ:* việc tốt nghiệp (cuối khóa) không thể làm thay đổi điểm đầu vào (lúc nhập học).
*Trong dự án:* cấm mọi mũi tên đi ra từ Y, và cấm T → Z.

**Mediator (biến trung gian)**
"Cây cầu" mà qua đó T tác động lên Y.
*Trong dự án:* học bổng → đỗ thêm tín chỉ HK1 → dễ tốt nghiệp hơn. **Không** được đưa biến này vào tập điều chỉnh, vì làm vậy sẽ "chặn" mất một phần hiệu ứng thật.

**Collider**
Biến bị hai thứ cùng gây ra (A → C ← B). Lọc dữ liệu theo C có thể tạo ra tương quan giả giữa A và B.
*Ví dụ:* nếu chỉ xét người được nhận vào trường danh tiếng (C), thì "giỏi thể thao" và "học giỏi" (A, B) có thể trông như ngược chiều nhau, dù ngoài đời chúng không liên quan.

---

## Phần 3 — Quan sát và can thiệp (cốt lõi của môn học)

**P(Y | T) — xác suất quan sát**
"Trong những người **đã có** học bổng, bao nhiêu % tốt nghiệp?" Câu này bị lẫn với việc nhóm có học bổng vốn đã khác nhóm không có.

**P(Y | do(T)) — xác suất can thiệp**
"Nếu nhà trường **phát** học bổng cho tất cả, bao nhiêu % sẽ tốt nghiệp?" Đây mới là câu hỏi chính sách.

**Graph surgery (phẫu thuật đồ thị)**
Hình dung `do(T)` là cắt hết các mũi tên đi **vào** T: học bổng giờ do nhà trường quyết định, không còn phụ thuộc điểm đầu vào hay hoàn cảnh nữa.
*Trong dự án:* Hình 6.1(b), các mũi tên màu đỏ nét đứt.

**Naive gap (chênh lệch thô)**
Hiệu đơn giản: 86.17% − 51.63% = **+34.54%**. Con số này bị phóng đại vì Z.

**Backdoor path (đường cửa sau)**
Đường "lén" nối T với Y thông qua Z (T ← Z → Y). Chính các đường này làm tương quan bị lẫn.

**Backdoor adjustment (điều chỉnh cửa sau)**
So sánh có học bổng và không có học bổng **trong từng nhóm Z giống nhau**, rồi lấy trung bình. Giống việc so 2 học sinh có cùng điểm vào, cùng hoàn cảnh, chỉ khác nhau ở học bổng.

**Identification (định danh)**
Câu hỏi: "Với DAG này, có **tính được** P(Y | do(T)) từ dữ liệu quan sát không?" Nếu được thì gọi là định danh được.
*Trong dự án:* DoWhy kiểm tra trên **từng DAG** mà mô hình học ra; kết quả đều định danh được bằng cửa sau.

---

## Phần 4 — Ba mô hình B0, H1, H2

| | Hiểu nhanh | Đồ thị | Phần "học sâu" |
| :--- | :--- | :--- | :--- |
| **B0** | Mạng Bayes truyền thống | Tự học bằng **HillClimb** trên dữ liệu TRAIN | Không có — bảng xác suất đếm tần suất |
| **H1** | B0 + mạng nơ-ron | **Dùng lại DAG của B0** | Bảng xác suất của Y được thay bằng MLP |
| **H2** | Học đồ thị bằng mạng nơ-ron | **NOTEARS-MLP** tự học | Mạng nơ-ron quyết định có cạnh nào |

*Vì sao thiết kế như vậy:* H1 so với B0 → chỉ khác ở cách tính xác suất. H2 so với B0 → chỉ khác ở cách học đồ thị. Mỗi so sánh đổi đúng **một** thứ.

**CBN (Causal Bayesian Network)**
DAG + bảng xác suất cho từng nút. Đủ để vừa dự báo, vừa trả lời câu hỏi "nếu can thiệp thì sao".

**CPD (bảng xác suất có điều kiện)**
**Bảng tra xác suất** của một nút khi đã biết giá trị các nút cha (các nút có mũi tên đi vào nó). Mỗi nút trong mạng Bayes có **một bảng**: sơ đồ mũi tên cho biết *cái gì ảnh hưởng cái gì*, còn CPD cho biết *ảnh hưởng bao nhiêu*.
*Ví dụ đời thường:* bảng tra phí bảo hiểm — biết tuổi và có hút thuốc không thì tra ra mức rủi ro.
*Trong dự án:* CPD của Y (B0, seed 42) có cha là học bổng, Debtor, học phí đúng hạn, tuổi > 20 → 2⁴ = 16 tình huống. Một dòng thật: "không nợ + đóng đúng hạn + ≤ 20 tuổi" → **73.6%** tốt nghiệp nếu không học bổng, **91.3%** nếu có học bổng. ATE được tính bằng cách so hai cột "có / không học bổng" của chính bảng này cho từng sinh viên TEST.
*Liên quan:* B0 điền bảng bằng BDeu; H1 chỉ thay **riêng bảng của Y** bằng mạng nơ-ron; H2 học sơ đồ khác rồi điền bảng bằng BDeu như B0.
*Nói với thầy:* "CPD là bảng xác suất của một nút khi biết các nút cha; mạng Bayes = sơ đồ + một CPD cho mỗi nút."

**HillClimb (leo đồi)**
Thử thêm, bớt hoặc đảo từng mũi tên; mũi tên nào làm điểm số tốt hơn thì giữ lại; lặp đến khi không cải thiện được nữa.

**BIC / BDeu / K2**
Ba cách "chấm điểm" một đồ thị: vừa khớp dữ liệu, vừa không quá phức tạp.
*Trong dự án:* thử cả ba, chọn cái tốt nhất trên VALIDATION → **K2** thắng ở cả 5 lần chạy.

**BDeu smoothing (làm mịn)**
Cộng thêm vài "mẫu ảo" vào mỗi ô của bảng để không có ô nào bằng 0% hay 100% tuyệt đối chỉ vì thiếu dữ liệu.
*Ví dụ (Slide 66, làm mịn cộng 1):* 5/5 học sinh đỗ ⇒ không kết luận là 100%, mà (5+1)/(5+2) ≈ 86%.
*Trong dự án:* ESS = 5 (độ mạnh của mẫu ảo).

**ESS (Equivalent Sample Size — cỡ mẫu tương đương)**
**Tổng số "người ảo"** mà BDeu cộng thêm khi điền bảng CPD. Người ảo được **chia đều** cho mọi ô và mọi kết cục, mang ý nghĩa "khi chưa biết gì thì coi như 50/50".
- ESS **nhỏ** → gần như chỉ tin dữ liệu thật (giống MLE).
- ESS **lớn** → kéo xác suất về 50/50 mạnh hơn.

*Ví dụ đời thường:* quán ăn cộng sẵn vài khách ảo "nửa khen nửa chê" cho mọi món. Món mới chỉ có 2 khách khen thì không bị ghi 100%; món đã có 1,000 khách thì vài khách ảo không đổi gì.
*Ví dụ bằng số (ô giả định có 4/4 người tốt nghiệp):*

| ESS | Xác suất tốt nghiệp của ô |
| :--- | :--- |
| 0 (đếm thô, MLE) | 4/4 = **100%** |
| 5 (dự án dùng) | ≈ **96%** |
| 20 | ≈ **88%** |
| 100 | ≈ **69%** |

Ô đông người (900/1000) thì ESS = 5 gần như không đổi gì (vẫn ≈ 90%) → làm mịn **chỉ tác động lên ô hiếm**.
*Trong dự án:* ESS = 5 cho B0 và H2 (rất nhỏ so với 2,541 SV trong TRAIN). Với bảng Y có 16 ô: mỗi ô nhận 5/16 ≈ 0.31 người ảo, mỗi bên tốt nghiệp / bỏ học ≈ 0.16 người. Quét ESS = 1, 5, 20 (script phụ, trên Domain DAG): ATE ≈ +0.237 → +0.238, gần như không đổi.
*Nói với thầy:* "ESS là số mẫu ảo của tiên nghiệm BDeu; nhóm dùng 5 — đủ chặn xác suất 0/1 ở ô hiếm mà không đè lên dữ liệu thật; đổi từ 1 đến 20 thì ATE chỉ lệch khoảng 0.001."

> **CPD và ESS liên quan thế nào?** **CPD là cái bảng; ESS là số người ảo cộng thêm khi điền số vào bảng.** Không có ESS, ô ít dữ liệu dễ ra 0% hoặc 100% vô lý; có ESS nhỏ (= 5), các ô hiếm được kéo về mức hợp lý còn các ô đông người gần như giữ nguyên.

**MLE**
Chỉ đếm tần suất thô, không làm mịn. Gặp ô ít dữ liệu dễ ra 0% hoặc 100% phi lý.

**MLP (mạng nơ-ron nhiều lớp)**
Hàm học được, nhận các nút cha của Y và trả về xác suất tốt nghiệp.
*Trong dự án:* 4 đầu vào, 769 tham số.

**Early stopping (dừng sớm)**
Ngừng huấn luyện khi điểm trên VALIDATION không còn tốt lên, để tránh "học vẹt".

**Dropout**
Mỗi lần học, tắt ngẫu nhiên 20% nơ-ron để mạng không phụ thuộc vào vài nơ-ron cụ thể.

**NOTEARS**
Biến việc "chọn mũi tên" thành bài toán tối ưu trơn: mỗi mũi tên có một trọng số, mạng điều chỉnh dần các trọng số, kèm một điều kiện phạt khi đồ thị có vòng.

**h(W) — điều kiện không có vòng**
Một con số bằng 0 khi và chỉ khi đồ thị không có vòng tròn.
*Trong dự án:* đạt khoảng 0.0000001 (gần như 0) ở cả 5 lần chạy ⇒ đã hội tụ.

**Mặt nạ M**
Danh sách mũi tên bị **cấm tuyệt đối** (ví dụ Y → bất cứ thứ gì). Trọng số của các mũi tên này luôn bằng 0.

**Ngưỡng τ**
Trọng số nhỏ hơn τ thì coi như không có mũi tên.
*Trong dự án:* τ chọn trên VALIDATION (0.1 hoặc 0.2).

**Cạnh bắt buộc T → Y**
Mũi tên học bổng → tốt nghiệp được giữ cố định ở B0, vì đó là **giả thuyết cần đo**. Không có cạnh này thì hiệu ứng theo định nghĩa bằng 0. Ở H2 cạnh này **tự học ra được**.

---

## Phần 5 — Chia dữ liệu và chạy thí nghiệm

**TRAIN / VALIDATION / TEST**
- **TRAIN (70%)** — học bài.
- **VALIDATION (15%)** — làm đề thử để chọn cách học.
- **TEST (15%)** — thi thật, chỉ mở sau khi đã chốt mọi lựa chọn.

**5 paired runs (5 lần chạy ghép cặp)**
Chia dữ liệu theo 5 cách khác nhau (seed 42–46). Trong mỗi cách, B0, H1, H2 dùng **đúng cùng** dữ liệu ⇒ so sánh công bằng.

**C_test (quần thể tham chiếu)**
545 sinh viên của tập TEST, dùng làm "dân số chung" để tính hiệu ứng cho cả ba mô hình.

**Không dùng Y_test để tính hiệu ứng**
Kết quả thật của sinh viên TEST chỉ dùng để chấm điểm **dự báo**. Hiệu ứng can thiệp chỉ dùng hồ sơ nền (Z) của họ.

---

## Phần 6 — Đo hiệu ứng

**ATE (hiệu ứng trung bình)**
Nếu **tất cả** đều có học bổng so với **không ai** có học bổng, tỷ lệ tốt nghiệp chênh bao nhiêu.
*Trong dự án:* B0 **+24.2%**, H1 **+23.7%**, H2 **+23.6%**.

**q_m(t, C_i)**
Xác suất tốt nghiệp **dự đoán** cho một sinh viên cụ thể nếu cho (t = 1) hoặc không cho (t = 0) học bổng.
*Trong dự án:* bảng cuối mục 8 của notebook in giá trị này cho vài sinh viên mẫu.

**risk(t)**
Trung bình q_m(t, C_i) trên cả 545 sinh viên TEST. ATE = risk(1) − risk(0).

**Khoảng tin cậy 95% (CI)**
Vùng giá trị hợp lý của ATE, tính bằng cách rút lại 1000 lần ngẫu nhiên các sinh viên trong TEST.
*Lưu ý:* cách này giữ mô hình cố định nên CI khá hẹp (±0.008); nó chưa tính độ không chắc chắn của chính đồ thị.

**Overlap (hỗ trợ)**
Mỗi kiểu sinh viên đều phải có người có và người không có học bổng; nếu không, mô hình đang phải đoán mò cho kiểu sinh viên đó.
*Trong dự án:* H2 có 1–2% sinh viên TEST rơi vào trường hợp thiếu hỗ trợ.

**G-computation / LPM / T-learner / IPW / AIPW**
Các cách khác để ước lượng ATE (dùng để đối chiếu):
- **LPM** — hồi quy tuyến tính.
- **G-computation** — dự đoán cho mọi người ở cả T = 1 và T = 0, rồi lấy trung bình hiệu.
- **T-learner** — hai mô hình riêng cho nhóm có và không có học bổng.
- **IPW** — cân trọng số theo khả năng được nhận học bổng.
- **AIPW** — kết hợp G-computation và IPW, đúng nếu ít nhất một trong hai mô hình đúng.

---

## Phần 7 — Chấm điểm dự báo (Table A)

**AUROC**
Xác suất mô hình chấm một sinh viên tốt nghiệp cao hơn một sinh viên bỏ học. 0.5 là đoán bừa; 1.0 là hoàn hảo.
*Trong dự án:* khoảng 0.80 ở cả ba mô hình.

**Macro-F1**
Điểm cân bằng giữa "bắt đúng" và "không bắt nhầm", tính trung bình đều cho cả hai lớp.

**Brier score**
Sai số bình phương giữa xác suất dự đoán và kết quả thật. Càng nhỏ càng tốt.

**Null finding (không tìm thấy khác biệt)**
H1 và H2 **không** dự báo tốt hơn B0 một cách đáng kể: chênh lệch nhỏ hơn mức dao động giữa các lần chạy. Báo cáo trung thực kết quả này là đúng yêu cầu môn học.

**Dự báo tốt ≠ nhân quả đúng**
Mô hình đoán Y rất giỏi vẫn có thể ra ATE sai, nếu DAG sai hoặc có biến gây nhiễu ẩn.

---

## Phần 8 — Kiểm tra độ tin cậy (Table C, mục 9)

**Refuter (bài kiểm tra bác bỏ)**
Cố tình "phá" dữ liệu để xem kết quả có phản ứng đúng như kỳ vọng không.

**Placebo refuter (thuốc giả)**
Xáo trộn ngẫu nhiên ai có học bổng. Hiệu ứng **phải về gần 0**.
*Trong dự án:* +0.0016 ✓.

**Subset refuter (tập con)**
Bỏ ngẫu nhiên 20% dữ liệu rồi tính lại. Hiệu ứng **phải gần như giữ nguyên**.
*Trong dự án:* 0.2309 → 0.2265 ✓.

**Random common cause**
Thêm một biến ngẫu nhiên vô nghĩa vào mô hình. Hiệu ứng **phải không đổi**.

**Kiểm tra huấn luyện lại (cho H1)**
Refuter của DoWhy không kiểm tra được mạng nơ-ron, nên nhóm tự làm lại placebo và subset bằng cách huấn luyện lại MLP.

**Vượt qua refuter không có nghĩa là DAG đúng**
Refuter chỉ cho thấy ước lượng ổn định; nó không chứng minh là không còn biến gây nhiễu ẩn.

**RV (Robustness Value — độ bền trước nhiễu ẩn)**
"Phải có một yếu tố **chưa đo** mạnh cỡ nào mới xóa được hiệu ứng?"
*Trong dự án:* **RV = 21%** — yếu tố ẩn đó phải giải thích được khoảng 21% phần biến thiên còn lại của **cả** việc nhận học bổng lẫn việc tốt nghiệp. Biến đo được mạnh nhất chỉ đạt khoảng 10%.

---

## Phần 9 — Phần mở rộng (phụ lục)

**CATE (hiệu ứng theo nhóm)**
Hiệu ứng tính riêng cho từng nhóm sinh viên.
*Trong dự án:* sinh viên trên 20 tuổi **+28%**, từ 20 tuổi trở xuống **+16%**.

**Relative risk (tỷ số rủi ro)**
Tỷ lệ tốt nghiệp khi có học bổng chia cho khi không có. Nhóm nền thấp có tỷ số lớn dù hiệu ứng tuyệt đối không lớn hơn.

**NDE / NIE (trực tiếp / gián tiếp)**
Tách hiệu ứng thành phần "đi thẳng" (NDE, +17%) và phần "đi qua cây cầu tín chỉ HK1" (NIE, +4.7%, chiếm khoảng 21%).

**T1 / T2 / T3**
- **T1** = học bổng (kết quả chính).
- **T2** = đóng học phí đúng hạn — đáng ngờ, vì có thể là **hậu quả** của việc bỏ học.
- **T3** = cả T1 và T2 cùng lúc; xấp xỉ T1 + T2, không có bằng chứng "cộng hưởng".

**Nhân quả ngược**
Tưởng A gây B nhưng thật ra B gây A.
*Ví dụ trong dự án:* sinh viên bỏ học thì ngừng đóng phí ⇒ "không đóng phí" trông như nguyên nhân bỏ học.

**Ablation (A1–A3)**
Bỏ đúng một thành phần để xem nó đóng góp bao nhiêu: A1 bỏ dropout, A2 bỏ mặt nạ, A3 bỏ làm mịn.

**OOD (Kaggle)**
Thử mô hình trên dữ liệu khác. Kaggle S4E6 là dữ liệu **tổng hợp**, và kết quả trái chiều ⇒ chưa chứng minh được gì về tính tổng quát.

---

## Phần 10 — 8 câu "thuộc lòng" khi bảo vệ

1. "Tương quan thô +34.5% bị phóng đại; sau khi điều chỉnh nhiễu, hiệu ứng còn khoảng **+24%**."
2. "B0 tự học DAG trên TRAIN bằng HillClimb, chọn hàm điểm trên VALIDATION; H1 giữ DAG của B0 và chỉ đổi bảng xác suất của Y; H2 học DAG bằng NOTEARS-MLP với cùng ràng buộc."
3. "Cả ba mô hình cho ATE gần như nhau ⇒ kết luận can thiệp ổn định."
4. "Phần nơ-ron không làm dự báo tốt hơn đáng kể — nhóm báo cáo đây là null finding."
5. "Hiệu ứng được tính trên hồ sơ nền của tập TEST; không dùng kết quả thật Y_test."
6. "Cạnh T → Y ở B0 là giả thuyết bắt buộc, không phải thứ học ra được."
7. "Refuter đều đạt, nhưng chỉ chứng tỏ ước lượng ổn định, không chứng minh DAG đúng; độ bền trước nhiễu ẩn RV = 21%."
8. "Hạn chế: biến học phí có thể là hậu quả của bỏ học; H2 thiếu hỗ trợ ở 1–2% hồ sơ; CI chưa tính độ không chắc chắn của đồ thị."
