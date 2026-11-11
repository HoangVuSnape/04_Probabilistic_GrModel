# Kịch bản nói — các cell cần trình bày

Bản chọn lọc **42 / 86 cell**, đi theo đúng thứ tự notebook (số cell giống notebook).
* **Mục 1–7 và 10 (kết luận):** 🎤 Nói, 📘 Từ khóa mới, ⚠️ Lưu ý.
* **Mục 8–9:** 🧩 Cell này làm gì → 🎤 Nói → 📊 Đọc output (đọc output khi chỉ vào màn hình).
* **Mục 9** có thêm phần **Tổng quan** ở đầu mục: mục tiêu, input, output và ý nghĩa của từng phần 9.A đến 9.H.

**Các cell được lướt qua, không cần nói riêng:**

| Cell | Lý do bỏ |
| :--- | :--- |
| 1, 5, 6, 7 | Tuyên bố dùng AI (đã có Vòng 0 trong QA), hướng dẫn cài đặt, in môi trường, đọc lại output cell 4 |
| 10, 15, 16, 19 | Markdown giới thiệu hoặc tóm tắt — nội dung đã nói khi trình bày cell code bên cạnh |
| 20, 21, 24, 26, 27, 28, 29, 32, 33, 34 | Phần lý thuyết dạng chữ và cell "đọc kết quả" nhắc lại output — gộp vào lời nói của cell hình/bảng cùng mục |
| 38, 39, 40, 42 | Mở đầu pipeline, nạp thư viện, mô tả dữ liệu 7 biến (đã có ở bảng từ điển biến) |
| 44, 46, 50, 51 | Markdown thiết kế B0 / H1 / giao thức đánh giá và hàm đánh giá dùng chung — đã nói trong cell code 45, 47, 52 |
| 54, 55, 56, 60, 63, 65 | Markdown đọc kết quả và cell xuất CSV — đã gộp vào cell 53, 58, 59, 61 |
| 66, 67, 68, 70, 72, 74, 76, 78, 80, 81, 82, 84 | Markdown mở đầu / đọc kết quả của mục 9 và bảng tổng hợp — đã gộp vào cell code tương ứng |

Notebook hiện tại đã bỏ phần phân bổ học bổng và chuyển sang Kaggle; mục **Tổng kết** là mục 10 (cell 85).
Câu hỏi chất vấn: tab **QA chất vấn**. Từ khóa khó: tab **Glossary** hoặc **Công thức**. Bản đủ từng cell: tab **Kịch bản đầy đủ** (đánh số theo bản notebook cũ 89 cell).


---

## MỞ ĐẦU

### Cell 0 · markdown · Tiêu đề và mục lục

**🎤 Nói:**
"Em xin trình bày đồ án môn Mô hình đồ thị xác suất. Câu hỏi của em rất đời thường: *nếu nhà trường cấp học bổng cho một sinh viên, khả năng sinh viên đó tốt nghiệp tăng thêm bao nhiêu?* Dữ liệu là bộ thật UCI 697, gồm hồ sơ sinh viên một trường đại học ở Bồ Đào Nha.

Toàn bộ đồ án nằm trong đúng một notebook này. Thầy bấm *Restart & Run All* thì mọi bảng, mọi hình và mọi đồ thị đều được sinh lại từ đầu, mất khoảng ba đến bốn phút trên CPU.

Về trình tự, em đi theo đúng mục lục. Mục 1 là dữ liệu và câu hỏi nghiên cứu. Mục 2 đến 7 là nền tảng lý thuyết. Mục 8 là phần chính: ba mô hình B0, H1, H2. Mục 9 là phần mở rộng. Mục 10 trả lời lại từng câu hỏi."

**📘 Từ khóa mới** *(mục lục nhắc rất nhiều từ; ở đây chỉ nói một câu cho mỗi từ, phần kỹ hơn nằm ở cell ghi trong ngoặc)*:
* **PGM — mô hình đồ thị xác suất:** biểu diễn một phân phối nhiều biến bằng đồ thị, trong đó nút là biến và cạnh là quan hệ phụ thuộc.
* **B0 / H1 / H2:** ba mô hình được so sánh. B0 là mạng Bayes truyền thống. H1 giữ đồ thị của B0 nhưng thay bảng xác suất của $Y$ bằng mạng nơ-ron. H2 dùng mạng nơ-ron để *học luôn đồ thị* (cell 37).
* **Tables A–C:** ba bảng kết quả bắt buộc. A là dự báo, B là can thiệp, C là độ vững và chi phí (cell 50, 54).
* **CATE:** hiệu ứng của học bổng *trong từng nhóm* sinh viên, ví dụ nhóm trên 20 tuổi (cell 78).
* **Trung gian (mediation):** tách xem học bổng tác động *trực tiếp*, hay *gián tiếp* qua kết quả học kỳ 1 (cell 76).


---

## MỤC 1 — LÝ DO, CÂU HỎI, DỮ LIỆU

### Cell 2 · markdown · 1.1a Lý do lựa chọn

**🎤 Nói:**
"Bảng này trả lời câu hỏi *vì sao em chọn như vậy*.

Dữ liệu UCI 697 là dữ liệu hành chính thật, công khai. Biến can thiệp ở đây do nhà trường quyết định, và bộ dữ liệu có kết cục cuối khóa.

Em chọn $T$ là học bổng vì đây là thứ nhà trường *thực sự thay đổi được*: có thể cấp hoặc không cấp. Ngược lại, tuổi hay giới tính thì không ai can thiệp được.

$Y$ là tốt nghiệp so với bỏ học. Em loại 794 sinh viên còn đang học vì họ chưa có kết cục.

Năm biến nền là những biến đo lúc nhập học, ảnh hưởng tới *cả* việc nhận học bổng *lẫn* khả năng tốt nghiệp.

Còn B0, H1, H2 được thiết kế để mỗi lần so sánh chỉ khác nhau đúng *một* yếu tố. H1 so với B0 chỉ khác cách tham số hóa. H2 so với B0 chỉ khác cách học đồ thị."

**📘 Từ khóa mới:**
* **Can thiệp hành động được (actionable):** biến mà ngoài đời có người thực sự bật/tắt được. Chỉ với biến như vậy thì câu hỏi "nếu cấp thì sao" mới có nghĩa chính sách.
* **$do(T=1)$:** ký hiệu của Pearl cho việc *ép* $T$ bằng 1 từ bên ngoài, khác với việc chỉ *quan sát thấy* $T=1$ (giải thích kỹ ở cell 16).
* **Biến nhiễu (confounder) $\mathbf{Z}$:** biến tác động lên *cả* $T$ và $Y$. Nếu không điều chỉnh theo nó, ta sẽ nhầm tương quan thành nhân quả.
* **Biến trung gian:** biến nằm *giữa* $T$ và $Y$ trên đường nhân quả, ví dụ tín chỉ HK1. Không được điều chỉnh theo nó khi đo *tổng* hiệu ứng.
* **CBN — Causal Bayesian Network:** mạng Bayes mà mỗi cạnh được hiểu là quan hệ nhân quả, nên có thể trả lời câu hỏi can thiệp.
* **HillClimb, BIC/BDeu/K2:** HillClimb là thuật toán leo đồi: thêm, xóa hoặc đảo cạnh sao cho điểm tăng. BIC, BDeu, K2 là ba *hàm điểm* khác nhau dùng để chấm một đồ thị.
* **TRAIN / VALIDATION / TEST:** ba phần dữ liệu tách rời. TRAIN để học. VALIDATION để chọn thiết lập. TEST để chấm cuối cùng.
* **Refuter, Cinelli–Hazlett:** refuter là các phép thử "phá" kết quả. Cinelli–Hazlett là phân tích độ nhạy trước *nhiễu ẩn* (cell 34, 80).

### Cell 3 · markdown · 1.1b Câu hỏi nghiên cứu

**🎤 Nói:**
"Câu hỏi chính RQ0 được viết thành một đại lượng cụ thể, gọi là ATE. Ngoài ra có năm câu phụ:
* RQ1: chênh lệch thô có bị thổi phồng không?
* RQ2 và RQ3: phần học sâu, H1 và H2, có giúp gì không?
* RQ4: kết luận vững tới đâu?
* RQ5 là phần mở rộng: học bổng tác động qua kênh nào và nhóm nào hưởng lợi.

Cột cuối của bảng ghi rõ chỗ trả lời từng câu. Ở mục 10, em quay lại bảng này và trả lời từng dòng."

**📘 Từ khóa mới:**
* **ATE — Average Treatment Effect, hiệu ứng can thiệp trung bình:**
  $$\text{ATE} = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]$$
  Đọc là: *"tỷ lệ tốt nghiệp nếu cấp học bổng cho **tất cả**, trừ đi tỷ lệ tốt nghiệp nếu **không cấp cho ai**."* Vì $Y$ là 0/1 nên ATE là chênh lệch xác suất. ATE = 0.24 nghĩa là tăng 24 **điểm phần trăm**.
* **Null finding:** kết quả "không thấy khác biệt". Đây là một phát hiện hợp lệ, không phải thất bại.

### Cell 4 · code · Nạp và kiểm toán dữ liệu

**🎤 Nói:**
"Cell này đọc file gốc rồi in thống kê. Bộ gốc có 4,424 dòng và 37 cột. Sau khi loại nhóm Enrolled, quần thể phân tích còn **3,630** sinh viên:
* 2,209 người tốt nghiệp (60.85%) và 1,421 người bỏ học.
* 969 người có học bổng (26.69%) và 2,661 người không có."

**⚠️ Lưu ý:** nếu thầy hỏi vì sao loại Enrolled, xem Vòng 1b trong file QA. Câu trả lời ngắn: *dữ liệu bị cắt phải* (right-censoring). Ngoài ra, mục 9.D đã giữ cả 4,424 sinh viên để kiểm tra, và kết quả vẫn dương: +16.87%.

### Cell 8 · markdown · 1.5 Từ điển biến và bảng giả định

**🎤 Nói:**
"Đây là bảng em coi là *hợp đồng* của đồ án. Mỗi biến có bốn thông tin: cột gốc trong dữ liệu, cách mã hóa, thời điểm đo, và vai trò.

Em đánh dấu thẳng một điểm yếu: biến *học phí đóng đúng hạn* có thời điểm đo **mơ hồ**, vì nó có thể được cập nhật sau khi nhập học. Vì vậy em ghi nó là *giả định*.

Bảng thứ hai liệt kê các giả định nhân quả, và cách em kiểm tra từng cái. Riêng giả định *không có nhiễu ẩn* thì không kiểm chứng được. Em chỉ có thể đo xem kết luận *nhạy tới đâu* nếu giả định này sai, bằng chỉ số RV ở mục 9."

**📘 Từ khóa mới:**
* **Estimand:** đại lượng ta *muốn* ước lượng, được định nghĩa trước khi chọn phương pháp. Ở đây estimand là ATE chuẩn hóa trên $C_{\text{test}}$.
* **Nhiễu ẩn (unobserved confounder):** một biến ảnh hưởng tới cả $T$ và $Y$ mà dữ liệu không có, ví dụ động lực học tập. Đây là kẻ thù số một của suy luận nhân quả từ dữ liệu quan sát.
* **Consistency:** giả định "học bổng" chỉ có *một phiên bản*. Nếu thực tế có nhiều mức học bổng khác nhau thì $do(T=1)$ không còn rõ nghĩa.
* **$RV_{q=1}$:** Robustness Value, giải thích kỹ ở cell 82.

### Cell 9 · code · Luồng loại trừ

**🎤 Nói:**
"Bảng này ghi lại từng bước lọc dữ liệu: 4,424 dòng, trừ 794 Enrolled, còn 3,630. Em kiểm tra thêm và thấy không có dòng trùng, không có giá trị thiếu ở các biến dùng trong mô hình, và điểm đầu vào nằm trong miền hợp lệ. Như vậy em không phải xóa hay điền thêm dòng nào ngoài nhóm Enrolled."

### Cell 11 · code · Bảng 1.6a — phân loại theo thời điểm đo

**🎤 Nói:**
"Trong 36 biến, có 24 biến đo trước hoặc lúc nhập học. Những biến này *được phép* làm biến nhiễu. 12 biến còn lại là tín chỉ và điểm của học kỳ 1, học kỳ 2. Chúng xảy ra *sau* học bổng nên bị loại khỏi tập điều chỉnh. Riêng số tín chỉ đỗ HK1 được giữ lại để làm biến trung gian ở mục 9."

**📘 Từ khóa mới:**
* **Biến hậu can thiệp (post-treatment):** biến đo sau $T$. Nếu đưa nó vào tập điều chỉnh, ta sẽ "cắt" mất một phần hiệu ứng thật đi qua nó, hoặc mở ra một đường giả.

### Cell 12 · code · Hình 1.1 — phân bố điểm đầu vào

**🎤 Nói:**
"Hình này giúp kiểm tra hai ngưỡng 120 và 140 em dùng để chia điểm đầu vào. Hai ngưỡng này chia ra ba nhóm đều đủ mẫu: 1,075, 1,869 và 686 sinh viên.

Nhóm dưới 120 điểm tốt nghiệp thấp hơn rõ, 52.6%, so với khoảng 64% ở hai nhóm trên. Tỷ lệ nhận học bổng cũng khác nhau giữa các nhóm, từ 23% đến 30%.

Như vậy điểm đầu vào liên quan tới *cả* $T$ *lẫn* $Y$. Đó đúng là dấu hiệu của một biến nhiễu."

**📘 Từ khóa mới:**
* **Rời rạc hóa (discretization):** chia biến liên tục thành vài nhóm, vì mạng Bayes rời rạc cần bảng xác suất. Ngưỡng được **chốt trước** theo ý nghĩa thực tế, không dò trên dữ liệu.

### Cell 13 · code · Hình 1.2 — Mutual Information

**🎤 Nói:**
"Hình này xếp hạng các biến theo mức liên kết với kết cục. Đứng đầu là *học phí đúng hạn*, với 0.101, mạnh hơn cả học bổng (0.066).

Kết quả này là một tín hiệu cảnh báo. Biến học phí liên kết mạnh tới mức có thể nó là *hậu quả* của việc bỏ học: sinh viên bỏ học thì ngừng đóng phí. Em sẽ quay lại điểm này ở mục 8.2b.

Ngoài ra, `Course` và `Application_mode` cũng liên kết khá mạnh nhưng không nằm trong $\mathbf{Z}$. Em coi chúng là *ứng viên nhiễu ẩn*, và đánh giá gián tiếp bằng chỉ số RV."

**📘 Từ khóa mới:**
* **MI — Mutual Information (thông tin tương hỗ):** đo xem biết $X$ thì giảm được bao nhiêu bất định về $Y$:
  $$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$
  MI bằng 0 khi hai biến độc lập. MI **chỉ đo liên kết, không đo nhân quả**, nên chỉ dùng để sàng lọc.

### Cell 14 · code · Bảng 1.6b — overlap theo tầng

**🎤 Nói:**
"Đây là kiểm tra overlap thô. Em chia sinh viên thành 6 tầng theo điểm đầu vào và tuổi. Tầng nào cũng có cả người có và người không có học bổng. Tầng nhỏ nhất vẫn có 44 người có học bổng, nên ở mức chia này không có tầng nào thưa.

Bảng còn cho thấy một điều đáng chú ý: sinh viên trên 20 tuổi nhận học bổng ít hơn hẳn, khoảng 12–16%, so với 30–38% ở nhóm trẻ hơn. Nghĩa là tuổi liên quan mạnh tới việc được nhận học bổng."


---

## MỤC 2 — CHÊNH LỆCH QUAN SÁT THÔ

### Cell 17 · code · Tính chênh lệch thô

**🎤 Nói:**
"Đây là phép tính ngây thơ nhất. Sinh viên có học bổng tốt nghiệp 86.17%, sinh viên không có học bổng tốt nghiệp 51.63%. Chênh lệch thô là **+34.54 điểm phần trăm**. Con số này là điểm xuất phát. Cả đồ án là để bóc xem trong 34.54 điểm đó, bao nhiêu là nhân quả thật."

**📘 Từ khóa mới:**
* **Naive gap (chênh lệch thô):** $P(Y=1 \mid T=1) - P(Y=1 \mid T=0)$. Đây chỉ là liên kết, chưa điều chỉnh gì.

### Cell 18 · code · Hình 2.1

**🎤 Nói:**
"Hình này vẽ lại hai tỷ lệ vừa tính để thấy trực quan khoảng cách 34.54 điểm."


---

## MỤC 3 — HỌC CẤU TRÚC (MINH HỌA)

### Cell 22 · code · Hình 3.1

**🎤 Nói:**
"Hình đặt hai kết quả cạnh nhau. Cạnh màu đỏ là cạnh trái thứ tự thời gian $\mathbf{Z} \to T \to Y$.

Bên trái là học tự do: hai cạnh đỏ đều đi ra từ *Tốt nghiệp*, tức kết cục đi ngược về quá khứ.

Bên phải là khi đã cấm các cạnh đi ra từ $Y$: hai cạnh đó biến mất. Nhưng xuất hiện cạnh đỏ mới, `Học bổng → Debtor`, nghĩa là học bổng gây ra tình trạng nợ. Cạnh này vẫn sai, vì Debtor là biến nền, có trước học bổng."

**⚠️ Lưu ý:** nếu thầy hỏi *"vậy hình này có sai không?"* — hình **không sai**. Nó vẽ đúng những gì thuật toán học ra, và các cạnh đỏ chính là chỗ hình cố ý chỉ ra là sai.

### Cell 23 · markdown · 3.3 Đọc kết quả

**🎤 Nói:**
"Vì sao bên phải vẫn còn cạnh `Học bổng → Debtor`? Vì ở thử nghiệm này em mới cấm các cạnh đi ra từ $Y$, *chưa* cấm các cạnh từ học bổng quay về biến nền.

Bài học là ràng buộc phải phủ **đủ** thứ tự thời gian: biến nền, rồi học bổng, rồi tốt nghiệp. Thiếu một tầng thì thuật toán vẫn có thể học sai chiều.

Vì vậy, khi học DAG thật cho B0 ở mục 8.B, em cấm cả hai loại: mọi cạnh đi ra từ $Y$, và mọi cạnh từ $T$ về $\mathbf{Z}$. H2 dùng đúng bộ ràng buộc này qua mặt nạ $M$.

Em cũng nói rõ vai trò của Domain DAG ở mục 6: nó **không thay thế** đồ thị học được. Nó chỉ dùng để minh họa DoWhy ở mục 7, và làm tham chiếu để so sánh với đồ thị mà B0 và H2 học ra. Mục 3 chỉ là minh họa trên 4 biến; mô hình thật nằm ở mục 8."

**⚠️ Lưu ý:**
* Nếu thầy hỏi *"vậy cuối cùng em có học DAG không, hay vẽ tay?"* — trả lời: **có học**. B0 dùng HillClimb, H2 dùng NOTEARS-MLP, đúng như Slide 57 yêu cầu. Đồ thị vẽ tay chỉ là tham chiếu và nguồn ràng buộc.
* Nếu thầy hỏi vì sao ở đây điểm vào chia thành 3 nhóm đều nhau chứ không theo ngưỡng 120/140: mục 3 chỉ minh họa nên chia nhanh theo tam phân vị; mô hình chính ở mục 8 dùng ngưỡng đã chốt ở mục 1.5.


---

## MỤC 4 — HỌC THAM SỐ: MLE VÀ BDEU

### Cell 25 · code · MLE và BDeu trên cùng một ô

**🎤 Nói:**
"Em đặt hai bảng CPD của $Y$ cạnh nhau, với cha là Debtor và học bổng. Ô *đang nợ học phí và có học bổng* có rất ít sinh viên. MLE cho xác suất tốt nghiệp khoảng 0.597. BDeu với ESS bằng 10 kéo con số này về gần mức chung hơn. Đó chính là tác dụng làm mịn."


---

## MỤC 5–6 — KIỂM TRA DAG VÀ DOMAIN DAG

### Cell 30 · code · Chạy bốn kiểm tra trên Domain DAG

**🎤 Nói:**
"Cả bốn kiểm tra đều đạt: không có chu trình, không có cạnh ngược thời gian, đúng ràng buộc chuyên gia, và 5 biến nền chặn hết đường cửa sau."

### Cell 31 · code · Hình 6.1

**🎤 Nói:**
"Hình 6.1 có hai panel. Cột bên trái là 5 biến nền, *Tốt nghiệp* nằm bên phải. Học bổng và tín chỉ HK1 được đặt ở hàng dưới, để thầy thấy rõ hai loại đường đi: các mũi tên ngang đi thẳng từ biến nền sang kết cục, và các mũi tên chéo xuống đi vào học bổng.

Panel (a) là đồ thị gốc, với các mũi tên từ biến nền đi vào học bổng. Panel (b) là đồ thị *sau phẫu thuật*: mọi mũi tên đi vào $T$ đã bị cắt (nét đứt màu đỏ), còn các mũi tên đi thẳng từ biến nền sang kết cục vẫn giữ nguyên."

**📘 Từ khóa mới:**
* **Graph surgery (phẫu thuật đồ thị):** cách Pearl mô tả phép $do(T=t)$. Ta cắt mọi cạnh *đi vào* $T$ rồi gán cố định $T=t$. Ý nghĩa: khi nhà trường *quyết định* cấp học bổng, học bổng không còn phụ thuộc vào điểm hay tuổi của sinh viên nữa.


---

## MỤC 7 — DOWHY

### Cell 35 · code · Chạy DoWhy

**🎤 Nói:**
"DoWhy tự tìm ra đúng tập 5 biến nền làm tập điều chỉnh cửa sau. Hai cách ước lượng cho kết quả gần nhau: hồi quy được **+24.12%**, trọng số propensity được **+25.16%**.

Ba refuter đều đạt:
* **Random common cause:** thêm một biến ngẫu nhiên làm nhiễu, ước lượng gần như không đổi (0.2412 thành 0.2411).
* **Placebo:** thay học bổng thật bằng một biến ngẫu nhiên, hiệu ứng rơi về 0.0047, gần bằng 0.
* **Data subset:** lấy ngẫu nhiên 80% dữ liệu, hiệu ứng vẫn là 0.2447."

**📘 Từ khóa mới:**
* **Refuter là gì, vì sao cần:** con số +24% do mô hình tính ra; câu hỏi là *nó có đáng tin không*, hay chỉ do may, do mô hình "bịa", hay do vài sinh viên đặc biệt. Ta không có "đáp án đúng" để so, nên thay vào đó **cố tình phá dữ liệu theo những cách mà ta biết trước kết quả phải ra sao**. Mô hình đáng tin thì phải phản ứng đúng như dự đoán. Ba refuter là ba cách phá khác nhau, mỗi cách bắt một loại lỗi:

* **Random common cause refuter (thêm biến nhiễu giả):**
  * *Làm gì:* thêm vào dữ liệu một cột **ngẫu nhiên hoàn toàn**, khai báo nó là một biến nhiễu ảnh hưởng cả học bổng lẫn tốt nghiệp, rồi ước lượng lại.
  * *Vì sao dùng:* một cột ngẫu nhiên không chứa thông tin gì, nên điều chỉnh theo nó **không được làm ATE đổi**. Nếu ATE nhảy, nghĩa là ước lượng **quá nhạy** — chỉ cần thêm một biến vô nghĩa đã đổi kết quả, thì kết quả đó không vững.
  * *Hình dung:* cân một món đồ, rồi đặt thêm một tờ giấy không trọng lượng lên cân — số không được đổi.
  * *Trong dự án:* $0.2412 \to 0.2411$ → **ổn định**.

* **Placebo treatment refuter (học bổng giả):**
  * *Làm gì:* **xáo trộn ngẫu nhiên** cột học bổng giữa các sinh viên (ai có học bổng là bốc thăm), giữ nguyên mọi thứ khác, rồi ước lượng lại.
  * *Vì sao dùng:* sau khi xáo, "học bổng" không còn liên quan gì tới sinh viên thật, nên **không thể có tác dụng** → hiệu ứng phải về **khoảng 0**. Nếu vẫn ra hiệu ứng lớn, nghĩa là mô hình đang **bắt được một mẫu hình giả** (ví dụ rò rỉ thông tin, hoặc cấu trúc dữ liệu tự sinh ra hiệu ứng) chứ không phải tác dụng thật của học bổng. Đây là phép thử mạnh nhất trong ba.
  * *Hình dung:* thử nghiệm thuốc với **viên thuốc đường** — nếu thuốc đường cũng "chữa khỏi" thì phương pháp đo có vấn đề.
  * *Trong dự án:* $0.2412 \to 0.0047 \approx 0$ → **đạt**.

* **Data subset refuter (bỏ bớt dữ liệu):**
  * *Làm gì:* chọn ngẫu nhiên **80%** dữ liệu, ước lượng lại; lặp nhiều lần.
  * *Vì sao dùng:* nếu hiệu ứng là thật thì nó phải xuất hiện ở **hầu hết các phần** của dữ liệu, không phụ thuộc vào vài sinh viên đặc biệt. Nếu bỏ 20% mà kết quả nhảy mạnh, nghĩa là kết luận **dựa vào một nhóm nhỏ** hoặc vài điểm ngoại lai.
  * *Hình dung:* khảo sát lại trên một nhóm người khác gần giống — kết luận đúng thì vẫn phải ra gần như cũ.
  * *Trong dự án:* $0.2412 \to 0.2447$ → **ổn định**.

* **Giới hạn chung:** cả ba chỉ kiểm tra **ước lượng có ổn định không** dưới các thay đổi biết trước. Chúng **không** chứng minh đồ thị đúng hay không có nhiễu ẩn — một biến chưa đo (như động lực học tập) vẫn có thể làm lệch kết quả mà cả ba refuter đều không phát hiện. Phần đó được đánh giá riêng bằng **Robustness Value** ở mục 9.H.

### Cell 36 · markdown · 7.4 Đọc kết quả

**🎤 Nói:**
"Con số 24.12% của DoWhy cùng bậc với kết quả chính ở Table B, khoảng 24%. Nhưng hai con số được tính khác nhau. DoWhy ở đây là hồi quy tuyến tính trên toàn bộ 3,630 sinh viên với biến gốc. Table B dùng CPD rời rạc và chuẩn hóa trên tập test.

Điều quan trọng nhất ở cell này là: **refuter chỉ kiểm tra độ ổn định của bộ ước lượng. Nó không chứng minh rằng không có nhiễu ẩn.** Phần nhiễu ẩn được xử lý bằng phân tích độ nhạy ở mục 9.H."

**📊 Đọc output (ba ý của cell 7.4 trong notebook):**

| Ý | Notebook ghi | Nói thế nào |
| :--- | :--- | :--- |
| **Kết quả** | DoWhy tự tìm đúng 5 biến nền cần điều chỉnh. Hiệu ứng **+24.12%** (hồi quy) và **+25.16%** (trọng số xu hướng), cùng bậc với Table B: B0 +24.18%, H1 +23.74%, H2 +23.58% | "Hai cách tính khác nhau ở DoWhy và ba mô hình ở mục 8 đều ra khoảng +24% — kết luận không phụ thuộc phương pháp." |
| **Vì sao số hơi khác Table B** | +24.12% là hồi quy tuyến tính trên **toàn bộ 3,630 SV** với biến gốc; Table B dùng **bảng xác suất rời rạc** trên **545 SV tập TEST** | "Khác dữ liệu và khác cách tính nên lệch nhau vài phần nghìn là bình thường." |
| **Ý nghĩa thực tế** | Cả 3 refuter đạt: thêm biến ngẫu nhiên $0.2412 \to 0.2411$; học bổng giả $\to +0.0047 \approx 0$; 80% dữ liệu $\to 0.2447$ | "Ước lượng không nhạy với biến vô nghĩa, không 'bịa' ra hiệu ứng khi học bổng là giả, và không phụ thuộc vào vài sinh viên." |
| **Tái lập** | Các refuter dùng `random_state=42` | "Chạy lại notebook sẽ ra đúng các con số này." |
| **Giới hạn** | DoWhy giả định không có nhiễu ẩn ngoài 5 biến đo được | "Câu hỏi 'nếu có biến chưa đo thì sao' em trả lời bằng phân tích độ nhạy Cinelli–Hazlett ở mục 9.H." |


---

## MỤC 8 — BA MÔ HÌNH B0, H1, H2 (PHẦN CHÍNH)

### Cell 37 · markdown · 8.1 Thiết kế B0 / H1 / H2 + 8.2 Multi-treatment

**🧩 Cell này là gì:** cell "bản đồ" của cả mục 8: mô tả ba mô hình bắt buộc và phần khám phá 8.2. Chưa có code chạy.

**🎤 Nói:**
"Mục 8 là phần chính của đồ án. Một mạng Bayes gồm hai phần: **sơ đồ mũi tên** và **bảng xác suất** cho từng nút. Ba mô hình khác nhau ở chỗ ai làm phần nào:
* **B0** — mạng Bayes truyền thống: HillClimb học sơ đồ trên TRAIN, BDeu điền bảng.
* **H1** — giữ nguyên sơ đồ của B0, chỉ thay **bảng của tốt nghiệp** bằng một mạng nơ-ron.
* **H2** — để mạng nơ-ron NOTEARS học **sơ đồ**, bảng vẫn điền bằng BDeu như B0.

Mỗi lần so sánh chỉ đổi đúng một thứ: H1 so với B0 là đổi cách tính xác suất; H2 so với B0 là đổi cách vẽ sơ đồ. Cả ba dùng cùng 7 biến, cùng dữ liệu, cùng luật cấm trong mỗi lần chạy.

Mục 8.2 là phần khám phá: ngoài học bổng, em thử thêm can thiệp *học phí đúng hạn* và *kết hợp cả hai*. Em giới thiệu ở đây, còn kết quả trình bày sau Tables A–C."

### Cell 41 · code · 8.A Config — mọi thiết lập ở một chỗ

**🧩 Cell này làm gì:** tạo một đối tượng `Config` "đóng băng" (không sửa được sau khi tạo) chứa **mọi** thiết lập của mục 8, và hàm `set_seed` cố định mọi nguồn ngẫu nhiên.

**🎤 Nói:**
"Toàn bộ thiết lập nằm ở một chỗ để không có con số nào 'ẩn' trong code. 5 seed 42–46, chia 70/15/15. B0 dùng ESS = 5 và thử ba hàm điểm. H1: 32 nơ-ron, dropout 0.2, tối đa 150 epoch, dừng sớm sau 15 epoch không tiến bộ. H2: lưới thử $\lambda_1$ và ngưỡng $\tau$, dừng khi ràng buộc vòng tròn dưới $10^{-6}$. Những gì có lưới thử đều được chọn trên VALIDATION."

**📊 Đọc output:** chỉ một dòng "Khởi tạo thành công…" — cell này không tính gì.

### Cell 43 · code · 8.A Năm cách chia dữ liệu dùng chung

**🧩 Cell này làm gì:**
1. Tạo "tầng" = tổ hợp (tốt nghiệp, học bổng), để mỗi phần có tỷ lệ giống nhau.
2. Chia 70% TRAIN / 30% còn lại, rồi chia đôi 30% thành VAL và TEST — mỗi lần đều giữ đúng tỷ lệ theo tầng.
3. Lưu chỉ số ra `artifacts/splits/split_seed_<s>.json`; lần sau đọc lại file thay vì chia mới (đóng băng).
4. Kiểm tra ba phần không trùng nhau và phủ đủ 3,630 dòng.

**🎤 Nói:**
"Mỗi seed cho một cách chia: 2,541 TRAIN, 544 VAL, 545 TEST. Chia phân tầng theo cả tốt nghiệp lẫn học bổng nên ba phần có tỷ lệ giống nhau. Trong mỗi seed, B0, H1, H2 dùng **đúng cùng** ba phần này — gọi là chạy ghép cặp."

**📊 Đọc output:** 5 dòng giống nhau (2541 / 544 / 545) — đúng, vì cùng tỷ lệ chia; chỉ khác *người nào* rơi vào phần nào.

### Cell 45 · code · 8.B B0 — học sơ đồ bằng HillClimb, chọn giám khảo trên VALIDATION

**🧩 Cell này làm gì (từng bước):**
1. **Đặt luật trước khi học:** cấm mọi mũi tên đi ra từ tốt nghiệp và mọi mũi tên học bổng → biến nền; bắt buộc học bổng → tốt nghiệp.
2. **Ba "giám khảo" chấm sơ đồ:** BIC, BDeu (ESS = 5), K2. Với mỗi giám khảo, HillClimb thử thêm / bớt / đảo từng mũi tên, giữ thay đổi làm điểm tăng → được 3 sơ đồ.
3. **Thi lại trên đề mới:** mỗi sơ đồ được điền bảng trên TRAIN, rồi đo xem đoán 544 sinh viên VAL tốt cỡ nào (log-likelihood, càng gần 0 càng tốt).
4. Chọn sơ đồ đoán VAL tốt nhất; kiểm tra không có vòng tròn và không có mũi tên cấm.
5. Định nghĩa hàm `query_model_effects`: tra bảng xác suất của tốt nghiệp để tính $q(0, C_i)$, $q(1, C_i)$ cho từng hồ sơ TEST. Hàm này dùng chung cho cả B0, H1, H2.
6. Chạy thử trên seed 42.

**🎤 Nói:**
"Ba giám khảo khác nhau ở mức khó tính với mũi tên mới: BIC phạt nặng, K2 dễ hơn. Không có lý thuyết nào nói trước giám khảo nào đúng, nên em để VALIDATION quyết định. Ở seed 42, BIC và BDeu cùng ra 9 mũi tên và bỏ cạnh Debtor → tốt nghiệp; K2 ra 11 mũi tên, giữ cạnh đó, và đoán VAL tốt hơn: −2208.37 so với −2211.11. K2 thắng ở cả 5 seed, nghĩa là Debtor thật sự giúp dự đoán. Cha của tốt nghiệp là học bổng, Debtor, học phí, tuổi. ATE trên TEST seed 42: 0.7863 − 0.5397 = **+0.2467**."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Bảng 3 ứng viên | Score = giám khảo; Edges = số mũi tên; Val_LogLik = điểm thi lại (gần 0 hơn là tốt hơn); Pa(Y) = cha của tốt nghiệp |
| "Chọn hàm điểm = K2 \| Cạnh học được …" | Sơ đồ cuối cùng của B0, seed 42 |
| Risk(do(T=0)) = 0.5397 | Nếu **không ai** có học bổng: 54% tốt nghiệp |
| Risk(do(T=1)) = 0.7863 | Nếu **ai cũng** có học bổng: 79% tốt nghiệp |

### Cell 47 · code · 8.C H1 — thay bảng xác suất của tốt nghiệp bằng mạng nơ-ron

**🧩 Cell này làm gì (từng bước):**
1. Lấy **đúng sơ đồ của B0**; đầu vào của mạng chỉ là **cha của tốt nghiệp** (học bổng, Debtor, học phí, tuổi — Slide 60).
2. Mạng: 4 đầu vào → 32 nơ-ron (BatchNorm, ReLU, Dropout 0.2) → 16 nơ-ron → 1 đầu ra; tổng **769 tham số**.
3. Huấn luyện trên TRAIN bằng BCE, Adam (lr 0.005), batch 64; sau mỗi epoch đo loss trên VAL; 15 epoch không tiến bộ thì dừng và lấy lại trọng số tốt nhất.
4. **Đổi mạng thành bảng:** liệt kê cả 16 tổ hợp cha, cho mạng đoán xác suất từng tổ hợp, ghi vào một bảng CPD.
5. Gắn bảng này vào bản sao của B0 (các bảng khác giữ nguyên), kiểm tra mô hình hợp lệ, rồi tính ATE bằng **cùng hàm** như B0.

**🎤 Nói:**
"H1 giữ nguyên sơ đồ của B0, chỉ thay bảng xác suất của tốt nghiệp bằng một mạng 769 tham số. Sau khi học, mạng được 'đổi ngược' thành bảng 16 ô và gắn lại vào mạng Bayes, nên cách tính hiệu ứng giống hệt B0 — chỉ khác cách ra con số trong bảng. Seed 42: ATE **+0.2453**, gần bằng B0."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Pa(Y) = [T, Z2, Z3, Z5] | Đầu vào = học bổng, Debtor, học phí, tuổi (4 biến nhị phân) |
| params = 769 | $(4 \times 32 + 32) + 64 + (32 \times 16 + 16) + (16 + 1)$ |
| best epoch = 2 | Đếm **từ 0**, tức là **epoch thứ 3** |
| ATE +0.2453 | Gần bằng B0 (+0.2467) |

### Cell 48 · markdown · 8.D Thiết kế H2

**🎤 Nói:**
"H2 dùng NOTEARS-MLP của Zheng và cộng sự (2020). Mỗi nút có một MLP nhỏ dự đoán nút đó từ các nút còn lại. Trọng số cạnh $W_{ij}$ là độ lớn của các trọng số lớp 1 đi từ biến $i$ vào MLP của nút $j$.

Mặt nạ $M$ được nhân thẳng vào trọng số, nên gradient ở các cạnh cấm bằng 0. Code còn `assert` rằng các ô bị cấm bằng đúng 0.

Bài toán được giải bằng Augmented Lagrangian. Các siêu tham số $\lambda_1$ và ngưỡng $\tau$ đều được chọn trên VALIDATION."

**🧒 Hiểu đơn giản — 7 ý của cell 8.D bằng lời thường:**

> **Hình dung chung:** H2 cho **mỗi biến một "học sinh"** (một mạng nơ-ron nhỏ). Học sinh của biến $j$ cố **đoán biến $j$** bằng cách "nghe" 6 biến còn lại. Học sinh nghe biến nào nhiều thì coi như có **mũi tên** từ biến đó sang $j$. Mạng nơ-ron ở đây chỉ dùng để **tìm mũi tên**, không dùng để dự báo.

| Ý trong cell | Nói đơn giản |
| :--- | :--- |
| **1. Mô hình** | Mỗi biến có một học sinh (10 nơ-ron). $W_{ij}$ = học sinh của $j$ "nghe" biến $i$ nhiều cỡ nào. $W_{ij} = 0$ nghĩa là không nghe, tức **không có mũi tên** $i \to j$. |
| **2. Mặt nạ $M$** | **Bịt tai trước** những gì không được nghe: học sinh của biến nền không được nghe Học bổng hay Tốt nghiệp (cấm ngược thời gian), và không ai được nghe chính mình. Bịt từ đầu nên **không thể** học ra mũi tên cấm; code còn kiểm tra lại các ô đó **đúng bằng 0**. Cùng danh sách cấm với B0. |
| **3. Mục tiêu** | Học sinh cố gắng ba điều cùng lúc: **đoán đúng** (sai số nhỏ) + **ít nghe lung tung** (phạt $\lambda_1$, giữ sơ đồ gọn) + **không được tạo vòng tròn** ($h(W) = 0$, "máy dò vòng tròn"). Vòng tròn bị phạt **ngày càng nặng** (×10 mỗi lần chưa giảm đủ) đến khi biến mất. |
| **4. Hội tụ** | Dừng khi máy dò vòng tròn báo gần 0 ($< 10^{-6}$). Cả **5/5 seed** đạt sau 7–8 vòng (khoảng $10^{-7}$). Nếu hết 12 vòng mà chưa đạt thì ghi rõ "không hội tụ" — thực tế không có. |
| **5. Chọn trên VALIDATION** | Hai núm vặn được chọn bằng dữ liệu chưa thấy: **$\lambda_1$** (phạt nghe lung tung mạnh cỡ nào) → chọn $10^{-4}$ ở seed 42 rồi giữ cố định; **$\tau$** (nghe mạnh cỡ nào mới tính là có mũi tên) → chọn riêng từng seed, thường là 0.1 hoặc 0.2. |
| **6. Sửa chu trình** | **Luật dự phòng, khai báo trước:** nếu sau khi lọc theo $\tau$ vẫn còn vòng tròn thì cắt mũi tên **yếu nhất** trong vòng; nếu thiếu Học bổng → Tốt nghiệp thì thêm vào như B0 và ghi rõ là bị ép. Thực tế **không seed nào cần** dùng luật này. |
| **7. Sau khi có sơ đồ** | **Bỏ mạng nơ-ron đi, chỉ giữ sơ đồ.** Bảng xác suất được điền bằng BDeu (ESS = 5) **y hệt B0**, và DoWhy kiểm tra lại sơ đồ có đo được hiệu ứng không. Vì vậy H2 so với B0 **chỉ khác đúng một thứ: sơ đồ mũi tên**. |

> **Câu một dòng:** "H2 giống B0, chỉ khác là sơ đồ mũi tên do mạng nơ-ron tìm ra (có bịt tai các mũi tên cấm và luật cấm vòng tròn), thay vì do HillClimb tìm ra."

**📘 Từ khóa mới:**
* **$h(W) = \operatorname{tr}(e^{W \circ W}) - d$:** vết của ma trận mũ đếm (có trọng số) các chu trình trong đồ thị. Hàm này bằng 0 khi và chỉ khi không có chu trình. Vì nó trơn, ta tối ưu được bằng gradient.
* **Augmented Lagrangian:** biến ràng buộc $h=0$ thành một số hạng phạt, rồi tăng dần mức phạt $\rho$ (nhân 10 mỗi lần) cho tới khi ràng buộc được thỏa.
* **$\lambda_1$ (phạt L1):** đẩy các trọng số nhỏ về 0, để đồ thị thưa hơn.
* **Ngưỡng $\tau$:** sau khi học xong, cạnh có $|W_{ij}| < \tau$ bị xóa.
* **Sửa chu trình:** nếu sau ngưỡng vẫn còn chu trình, cắt cạnh có trọng số nhỏ nhất trong chu trình. Luật này được khai báo trước khi chạy.

### Cell 49 · code · 8.D H2 — chọn $\lambda_1$, chọn ngưỡng $\tau$, kiểm tra trên seed 42

**🧩 Cell này làm gì (từng bước):**
1. Tạo **mặt nạ M** từ đúng danh sách cấm của B0, cộng thêm cấm tự nối.
2. Mạng NOTEARS-MLP: mỗi biến có một mạng nhỏ đoán biến đó từ các biến khác; độ mạnh mũi tên $W_{ij}$ lấy từ trọng số lớp đầu.
3. Tối ưu: đoán đúng + phạt nhiều mũi tên ($\lambda_1$) + phạt vòng tròn $h(W)$ tăng dần; mỗi vòng ngoài 300 bước, dừng khi $h < 10^{-6}$.
4. **Chọn $\lambda_1$** trong {$10^{-3}$, $10^{-4}$} trên VAL của seed 42, rồi giữ cố định.
5. **Chọn $\tau$** trong {0.1, 0.2, 0.3, 0.5, 0.8}: lọc mũi tên ≥ $\tau$, điền bảng BDeu trên TRAIN, chấm trên VAL.
6. Điền bảng BDeu như B0 và tính ATE.

**🎤 Nói:**
"Bảng đầu là chọn mức phạt số mũi tên: $10^{-4}$ thắng (−2251 so với −2277) nên giữ cho mọi seed. Bảng thứ hai là chọn ngưỡng: $\tau = 0.1$ giữ 8 mũi tên và đoán VAL tốt nhất. Thầy nhìn cột T->Y_forced: ở ngưỡng được chọn là False, nghĩa là mũi tên học bổng → tốt nghiệp được **tự học ra**, không phải ép; chỉ khi ngưỡng quá cao (0.8) chỉ còn 1 mũi tên mới phải ép. Ràng buộc vòng tròn $3.3 \times 10^{-7}$ — đã hội tụ; trọng số ở ô bị cấm đúng bằng 0. ATE **+0.2428**."

**📊 Đọc output:**
| Cột / dòng | Nghĩa |
| :--- | :--- |
| `lambda_l1`, `best_Val_LogLik` | Mức phạt số mũi tên và điểm thi lại tương ứng |
| `tau`, `Edges` | Ngưỡng và số mũi tên còn lại (0.1 → 8, 0.8 → 1) |
| `Cut_for_cycle` = [] | Không phải cắt mũi tên nào để phá vòng |
| `T->Y_forced` | True = phải ép thêm học bổng → tốt nghiệp |
| `Pa(Y)` ở τ = 0.1 | H2 thêm **điểm vào (Z1)** làm cha của tốt nghiệp, khác B0 |
| `max |W| ở ô bị cấm = 0.0` | Mặt nạ hoạt động đúng |

### Cell 52 · code · 8.E Vòng lặp 5 lần chạy ghép cặp

**🧩 Cell này làm gì:** với mỗi seed 42–46: lấy đúng split của seed đó → học B0 → học H1 trên sơ đồ B0 → học H2 → với mỗi mô hình gọi `record_model_run` để (a) chấm dự báo trên TEST bằng nhãn thật, (b) tính ATE trên hồ sơ TEST **không dùng nhãn**, (c) CI bootstrap, (d) chạy DoWhy identify trên sơ đồ của chính mô hình, (e) kiểm tra overlap. Mọi đồ thị, trọng số, checkpoint được lưu ra `artifacts/`.

**🎤 Nói:**
"Đây là vòng lặp chính. Mỗi dòng tóm tắt một seed: sơ đồ B0 luôn do K2 chọn, 10 đến 12 mũi tên; H2 luôn hội tụ, ngưỡng 0.1 hoặc 0.2. ATE của cả ba mô hình ở mọi seed đều nằm trong khoảng **0.22 đến 0.25**."

**📊 Đọc output:**
| Seed | B0 | H1 | H2 |
| :--- | :--- | :--- | :--- |
| 42 | +0.247 | +0.245 | +0.243 |
| 43 | +0.247 | +0.239 | +0.240 |
| 44 | +0.238 | +0.218 | +0.230 |
| 45 | +0.238 | +0.242 | +0.228 |
| 46 | +0.239 | +0.243 | +0.239 |

### Cell 53 · code · 8.E2 Quá trình huấn luyện, overfit, có / không early stopping

**🧩 Cell này làm gì:**
1. Vẽ **Hình 8.2**: (a) loss TRAIN/VAL của H1 qua từng epoch, chấm tại epoch tốt nhất; (b) $h(W)$ của H2 qua từng vòng (thang log); (c) sai số tái tạo của H2.
2. **Bảng 8.E2**: AUROC, accuracy, log-loss của B0/H1/H2 trên **cả** TRAIN, VAL, TEST.
3. Huấn luyện thêm bản H1 **không early stopping** (đủ 150 epoch), lưu 2 checkpoint riêng, so sánh (**Hình 8.2b, Bảng 8.E2b**).

**🎤 Nói:**
"Trước khi xem kết quả, em cho thầy thấy quá trình huấn luyện ổn. Loss VAL của H1 chạm đáy ở epoch 2–4 (seed 43 là 24) nên dừng sau 17–39 epoch. H2 đưa ràng buộc vòng tròn về khoảng $10^{-7}$ sau 7–8 vòng. AUROC trên TRAIN, VAL, TEST gần như bằng nhau — chênh TRAIN trừ TEST: B0 +0.002, H1 khoảng 0, H2 +0.008 — nên không overfit đáng kể. Bỏ early stopping thì loss VAL tăng (0.504 so với 0.500) và AUROC TEST giảm nhẹ — đúng dấu hiệu bắt đầu overfit."

**📊 Đọc output:**
| Chỗ | Nghĩa |
| :--- | :--- |
| Hình 8.2a, nét liền / nét đứt | Loss TRAIN / VAL; chấm tròn = trọng số được giữ |
| Hình 8.2b | $h(W)$ giảm đều theo thang log, xuống dưới vạch $10^{-6}$ |
| Hình 8.2c | Sai số tái tạo **tăng** nhẹ — cái giá của việc ép thành DAG |
| Cột `AUROC_gap_TRAIN−TEST` | Thước đo overfit; càng gần 0 càng tốt |
| Bảng 8.E2b | So hai bản H1; ATE trung bình gần như không đổi (0.237 / 0.236) |

### Cell 57 · code · 8.F Refuter và chi phí — tạo Table C

**🧩 Cell này làm gì:**
1. Trên **sơ đồ của B0** và **sơ đồ của H2** (seed 42, TRAIN): chạy DoWhy với ước lượng hồi quy tuyến tính, rồi 2 phép thử phá, mỗi phép 30 lần:
   * **Placebo:** xáo trộn ngẫu nhiên cột học bổng → hiệu ứng phải về khoảng 0.
   * **Subset:** chỉ dùng 80% dữ liệu → hiệu ứng phải gần như cũ.
2. **H1** không chạy được refuter tuyến tính, nên tự làm phép thử tương đương: xáo học bổng rồi **huấn luyện lại mạng**; và huấn luyện lại trên 80% TRAIN (3 lần).
3. Gom độ tản ATE giữa 5 seed và thời gian chạy thành Table C.

**🎤 Nói:**
"Phép placebo xáo trộn ngẫu nhiên ai có học bổng: nếu mô hình đúng, 'học bổng giả' không được có tác dụng — và thực tế về +0.002 cho B0, H2, +0.018 cho H1. Phép subset bỏ 20% dữ liệu: hiệu ứng vẫn khoảng 0.22–0.24. Cả ba mô hình đều qua."

**📊 Đọc output:** cell chỉ in "đã lưu"; kết quả nằm ở Table C (cell sau).

### Cell 58 · code · Hiển thị Tables A, B, C, H3, Multi-Treatment

**🧩 Cell này làm gì:** đọc các bảng CSV vừa lưu và hiển thị (số làm tròn 4 chữ số; CSV giữ đủ độ chính xác).

**🎤 Nói:**
"**Table A — dự báo trên TEST**: AUROC B0 0.800, H1 0.801, H2 0.803. Hiệu ghép cặp H1 trừ B0 là +0.002 ± 0.003, H2 trừ B0 là +0.004 ± 0.004 — nhỏ hơn độ dao động giữa các lần chạy, nên mạng nơ-ron **không cải thiện rõ** dự báo. Đây là null finding hợp lệ.

**Table B — can thiệp**: ATE B0 +0.242 ± 0.005, H1 +0.237 ± 0.011, H2 +0.236 ± 0.007. **Kết luận không đổi giữa ba mô hình.** So với 34.5 điểm thô, khoảng 10 điểm là thiên lệch do khác biệt sẵn có.

**Table C — độ vững**: độ tản ATE giữa 5 seed 0.009 (B0), 0.028 (H1), 0.015 (H2); thời gian B0 dưới 1 giây, H2 chậm nhất."

**📊 Đọc output:**
| Bảng / cột | Nghĩa |
| :--- | :--- |
| A · `Macro_F1` B0 = H1 | Bảng chỉ 16 tình huống nên hai mô hình gán nhãn giống hệt; khác nhau ở xác suất (AUROC, Brier) |
| A · `Delta_*_vs_B0` | Hiệu theo cặp trong cùng seed |
| B · `Adjustment_Set` | Biến dùng để điều chỉnh = cha của tốt nghiệp trừ học bổng: {Z2, Z3, Z5}; H2 thêm Z1 |
| B · `DoWhy_Backdoor_Set` = {Z3, Z5} | Tập **tối thiểu** DoWhy tìm được; Debtor không cần cho định danh vì đường qua nó đã bị chặn ở học phí, nhưng thêm vào giúp ước lượng chính xác hơn |
| B · `Overlap_Status` | H2: 5–12 hồ sơ TEST (1–2%) thiếu dữ liệu so sánh, ở đó dựa vào làm mịn BDeu |
| B · `CI_95` | Khoảng ±0.008 — chỉ phản ánh việc chọn 545 hồ sơ, mô hình giữ cố định |
| H3 | Phụ lục, script riêng, dùng Domain DAG cố định — không so số với A–C |
| Multi-Treatment | Học bổng +0.21 đến +0.23 qua 4 bộ ước lượng; học phí và kết hợp: khám phá (cell sau) |

### Cell 59 · code · 8.2b Positivity và dấu hiệu nhân quả ngược của học phí

**🧩 Cell này làm gì:** đếm số sinh viên và tỷ lệ tốt nghiệp trong 4 ô (học bổng × học phí đúng hạn).

**🎤 Nói:**
"Mục 8.2 thử can thiệp *học phí đúng hạn* và ra +45 đến +51% — quá lớn để tin. Bảng 4 ô giải thích: người không học bổng và chưa đóng đúng hạn chỉ **4.5%** tốt nghiệp — giống 'bỏ học rồi ngừng đóng phí' hơn là 'nợ phí nên bỏ học', tức có thể là nhân quả ngược. Còn ô vừa có học bổng vừa chưa đóng đúng hạn chỉ **37 sinh viên** — điều kiện dương yếu, ước lượng ở đó dựa nhiều vào ngoại suy. Vì vậy em chỉ kết luận về học bổng."

**📊 Đọc output:** 449 SV (4.5%) · 2,212 SV (61.2%) · **37 SV (24.3%)** · 932 SV (88.6%).

### Cell 61 · code · Hình 8.1 — B0 và H2 học được sơ đồ gì

**🧩 Cell này làm gì:** đọc lại sơ đồ đã lưu của 5 seed → bảng tóm tắt; vẽ Hình 8.1: (a) mặt nạ M, (b) ma trận $|W|$ của H2 (khung đỏ = mũi tên được giữ), (c) sơ đồ B0, (d) sơ đồ H2 cùng bố cục.

**🎤 Nói:**
"Bảng tóm tắt cho thấy B0 rất ổn định: K2 ở cả 5 seed, 10–12 mũi tên, cha của tốt nghiệp luôn là học bổng, Debtor, học phí, tuổi. H2 thưa hơn, 7–9 mũi tên, hội tụ mọi seed, không seed nào phải ép học bổng → tốt nghiệp, và ở 4/5 seed thêm điểm vào làm cha của tốt nghiệp.

Trên hình: B0 học thêm chuỗi mũi tên **giữa các biến nền**; H2 gần như chỉ giữ mũi tên vào học bổng và tốt nghiệp. Chiều giữa các biến nền đổi theo mô hình và seed — dữ liệu không xác định được chiều giữa các biến cùng thời điểm, nhưng điều đó không ảnh hưởng truy vấn về học bổng vì chúng đều có trước học bổng."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| `B0_score(val)` | Giám khảo được chọn (K2 cả 5 seed) |
| `H2_tau(val)` | Ngưỡng được chọn (0.1 / 0.2) |
| `H2_T->Y_forced` | False cả 5 seed → cạnh được tự học |
| `H2_max|W|_forbidden` | 0 cả 5 seed → mặt nạ tuân thủ |

### Cell 62 · code · Tính tay $q_m(t, C_i)$ để thấy công thức minh bạch

**🧩 Cell này làm gì:** dựng lại B0 và H1 seed 42 từ file đã lưu, chọn vài hồ sơ TEST, tự tra bảng / tự cho mạng đoán: xác suất tốt nghiệp nếu đặt học bổng = 0 và = 1. Rồi lấy trung bình trên 545 hồ sơ.

**🎤 Nói:**
"Cell này mở 'hộp đen': với mỗi hồ sơ, em đặt học bổng bằng 0 rồi bằng 1, giữ nguyên hoàn cảnh, và đọc xác suất tốt nghiệp. Ví dụ hồ sơ không nợ, đóng đúng hạn, dưới 20 tuổi: B0 cho 73.6% nếu không học bổng và 91.3% nếu có. Lấy trung bình 545 hồ sơ ra đúng ATE của Table B: +0.2467 cho B0, +0.2453 cho H1. Không dùng nhãn TEST ở bước nào."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| Z2, Z3, Z5 | Hoàn cảnh của hồ sơ (Debtor, học phí, tuổi) |
| `q_B0(0,C_i)`, `q_B0(1,C_i)` | Xác suất tốt nghiệp nếu không / có học bổng, theo B0 |
| `q_H1(…)` | Tương tự theo H1 — gần B0 vì cùng sơ đồ |
| Dòng risk / ATE | Trung bình trên 545 hồ sơ = công thức Slide 59, 61 |

### Cell 64 · code · Hình 8.3 — forest plot ATE theo seed

**🧩 Cell này làm gì:** vẽ ATE và CI 95% của từng seed cho B0, H1, H2, kèm vạch chênh lệch thô.

**🎤 Nói:**
"Mỗi chấm là ATE của một mô hình ở một seed, thanh ngang là CI. Mọi chấm đều quanh +0.24 và cách xa vạch xám +0.345 của chênh lệch thô. Nhìn một hình là thấy: dù mô hình nào, seed nào, kết luận vẫn như nhau."


---

## MỤC 9 — PHÂN TÍCH MỞ RỘNG

### Tổng quan mục 9 — mục tiêu, input, output, ý nghĩa từng phần

**Mục tiêu chung:** học bổng có làm được gì ngoài tỷ lệ tốt nghiệp? Mục này đo thêm kết cục khác, tìm đường đi của tác dụng, xem ai được lợi nhiều hơn, và kiểm tra độ vững.

#### 9.A: Chuẩn bị hai quần thể (Cell 69)

- **Mục tiêu:** tạo đúng bộ dữ liệu cho từng phép đo.
- **Input:** `data/uci_697.csv`, 4,424 SV. Biến cần dùng: học bổng, tốt nghiệp, điểm HK2, tín chỉ HK2, tín chỉ HK1, và 7 biến điều chỉnh (tuổi, điểm vào, mẹ có bằng, Debtor, học phí đúng hạn, displaced, giới tính).
- **Output:**
    - `df_completed`: 3,630 SV có kết cục, dùng cho 9.B, 9.C, 9.E, 9.F.
    - `df_full`: 4,424 SV, dùng cho 9.D.
- **Ý nghĩa:** giữ người đang học ở bước giữ chân để kiểm tra việc loại họ ở mục 8 có làm lệch kết luận không.

#### 9.B: Điểm HK2 (Cell 71)

- **Mục tiêu:** học bổng có làm điểm học kỳ 2 tăng không?
- **Input:** `df_completed`; kết cục là điểm HK2 (thang 20); học bổng và 7 biến điều chỉnh.
- **Làm gì:** hồi quy OLS: điểm ~ học bổng + 7 biến. Hệ số của học bổng là ATE.
- **Output:**
    - ATE = **+1.478 điểm**, CI 95% [1.093, 1.863], p ≈ 7×10⁻¹⁴.
    - Chênh lệch thô là +2.63, nên điều chỉnh làm giảm gần một nửa.
- **Ý nghĩa:** học bổng có tác dụng tích cực lên học tập. Tuy nhiên 22% SV có điểm bằng 0, và trong nhóm bỏ học tỷ lệ này là 51%, nên một phần kết quả phản ánh việc còn đi học.

#### 9.C: Tỷ lệ đạt tín chỉ HK2 (Cell 73)

- **Mục tiêu:** học bổng có giúp sinh viên đạt nhiều môn đăng ký hơn không?
- **Input:** `df_completed`; kết cục là tín chỉ đạt chia tín chỉ đăng ký (mẫu số tối thiểu 1); học bổng và 7 biến.
- **Làm gì:** hồi quy OLS như 9.B.
- **Output:**
    - ATE = **+14.50 điểm %**, CI [11.80%, 17.20%].
    - Tỷ lệ đạt 100% số môn đăng ký: 33.9% (không học bổng) so với 58.2% (có học bổng), số thô chưa điều chỉnh.
- **Ý nghĩa:** học bổng giảm việc nợ tín chỉ. Đây cũng là kết cục sau can thiệp nên không được dùng làm biến điều chỉnh.

#### 9.D: Giữ chân toàn trường bằng AIPW (Cell 75)

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

#### 9.E: Trung gian: học bổng đi qua đường nào? (Cell 77)

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

#### 9.F: Ai được lợi nhiều hơn? (CATE, Cell 79)

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

#### 9.G: Tổng hợp (Cell 81)

- **Mục tiêu:** gom kết quả 9.B đến 9.E vào một bảng.
- **Input:** kết quả của các phần trước.
- **Output:** `Table_Alternative_Outcomes.csv` và `Table_Heterogeneous_CATE.csv`.
- **Ý nghĩa:** bốn chiều kết quả đều dương, nên kết luận "học bổng có lợi" không phụ thuộc vào kết cục nào được chọn. Lưu ý: cột p-value trong bảng này ghi chữ cố định như "< 1e-10", không phải giá trị tính lại.

#### 9.H: Độ nhạy trước nhiễu ẩn (Cell 83)

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

#### Tóm tắt một dòng cho mỗi phần

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

### Cell 69 · code · 9.A Hai quần thể cho phần mở rộng

**🧩 Cell này làm gì:** tạo hai bảng dữ liệu riêng cho mục 9:
* **3,630 SV có kết cục** (như mục 8) — cho điểm HK2, tín chỉ, trung gian, CATE.
* **4,424 SV toàn trường** (kể cả đang học) với kết cục *giữ chân* = 1 nếu không bỏ học — cho mục 9.D.

Khác mục 8: dùng biến gốc (tuổi, điểm vào dạng số) và **7 hiệp biến** (thêm Displaced, Gender), ước lượng bằng hồi quy thay vì mạng Bayes.

**🎤 Nói:**
"Mục 9 hỏi rộng hơn: học bổng giúp ở những mặt nào, qua đường nào, giúp ai nhiều hơn, và kết luận có dễ bị lật không. Em dùng các công cụ thống kê nhân quả chuẩn, độc lập với mạng Bayes, để kiểm tra chéo."

### Cell 71 · code · 9.B Kết cục 1 — điểm trung bình HK2

**🧩 Cell này làm gì:** hồi quy tuyến tính: điểm HK2 ~ học bổng + 7 hiệp biến; hệ số của học bổng = ATE (thang 20).

**🎤 Nói:**
"Học bổng làm điểm HK2 tăng **+1.48 điểm** trên thang 20, CI 1.09 đến 1.86. Chênh lệch thô là 2.63, nên sau điều chỉnh còn khoảng một nửa — lại là dấu vết của khác biệt sẵn có."

**📊 Đọc output:** ATE +1.478; CI [1.093, 1.863]; p ≈ $10^{-13}$; $R^2$ = 0.17 (mô hình giải thích 17% độ biến thiên điểm — bình thường với dữ liệu người).

### Cell 73 · code · 9.C Kết cục 2 — tỷ lệ đạt tín chỉ HK2

**🧩 Cell này làm gì:** tỷ lệ = tín chỉ đạt / tín chỉ đăng ký ở HK2 (mẫu số tối thiểu 1 để tránh chia cho 0), rồi hồi quy như 9.B.

**🎤 Nói:**
"Tỷ lệ đạt tín chỉ tăng **+14.5 điểm %**, CI 11.8 đến 17.2. Tỷ lệ sinh viên đạt hết mọi môn đăng ký: 58% ở nhóm có học bổng so với 34% nhóm không — đây là số thô, chưa điều chỉnh."

### Cell 75 · code · 9.D Kết cục 3 — giữ chân toàn trường bằng AIPW

**🧩 Cell này làm gì (từng bước):**
1. **Mô hình xác suất được học bổng** $e(x)$: hồi quy logistic học bổng ~ hiệp biến; chặn trong [0.05, 0.95] để không có trọng số quá lớn.
2. **Hai mô hình kết cục:** logistic riêng cho nhóm có học bổng ($\mu_1$) và không có ($\mu_0$), đoán khả năng ở lại.
3. **Điểm AIPW** từng sinh viên: dự đoán + phần "sửa sai" bằng sai số thật, có trọng số theo $e(x)$.
4. ATE = trung bình điểm có học bổng − trung bình điểm không học bổng.
5. CI: bootstrap 1,000 lần trên các điểm AIPW.

**🎤 Nói:**
"Kết cục giữ chân tính trên **cả 4,424 sinh viên**, kể cả 794 người đang học — nên nó cũng trả lời câu hỏi việc loại Enrolled có làm chệch không. Không có học bổng: 64.3% ở lại; có học bổng: 81.2%; ATE **+16.87%**, CI 13.79 đến 19.73. AIPW 'chắc ăn gấp đôi': chỉ cần một trong hai mô hình — xác suất được học bổng hoặc kết cục — đúng là kết quả không chệch."

**📊 Đọc output:** Risk(do(T=0)) 64.28% · Risk(do(T=1)) 81.15% · ATE +16.87% · CI [13.79%, 19.73%].

### Cell 77 · code · 9.E Trung gian — học bổng giúp qua đường nào

**🧩 Cell này làm gì (phương pháp tích hệ số):**
1. Hồi quy **tín chỉ đạt HK1** ~ học bổng + hiệp biến → hệ số $a$: học bổng làm tăng bao nhiêu tín chỉ HK1.
2. Hồi quy **tốt nghiệp** ~ học bổng + tín chỉ HK1 + hiệp biến → hệ số của học bổng = **tác dụng trực tiếp (NDE)**; hệ số $b$ của tín chỉ.
3. **Tác dụng gián tiếp (NIE)** = $a \times b$; tỷ lệ trung gian = NIE / (NDE + NIE).

**🎤 Nói:**
"Học bổng làm sinh viên đạt thêm **0.73 tín chỉ** ở HK1; mỗi tín chỉ thêm lại tăng khả năng tốt nghiệp. Nhân lại: tác dụng đi qua HK1 là **+4.65%**, trực tiếp là **+16.96%**, tổng +21.61%. Vậy khoảng **21.5%** tác dụng của học bổng đi qua việc học tốt hơn ở học kỳ đầu; phần còn lại là trực tiếp — bớt lo tài chính, có thêm động lực."

**📊 Đọc output:** hệ số T → M +0.729 tín chỉ · TE +21.61% · NDE +16.96% (78.5%) · NIE +4.65% (21.5%).

### Cell 79 · code · 9.F CATE — học bổng giúp ai nhiều hơn

**🧩 Cell này làm gì:** với 6 nhóm định trước (tuổi, Debtor, nơi ở): trong từng nhóm, khớp logistic tốt nghiệp ~ học bổng + hiệp biến; đặt học bổng = 0 cho mọi người, đoán, lấy trung bình; đặt = 1, làm lại; hiệu = CATE (G-computation).

**🎤 Nói:**
"Sinh viên **trên 20 tuổi** được lợi nhiều nhất: +28.0 điểm, so với +16.1 ở nhóm đúng tuổi. Với nhóm có nợ, hiệu ứng tuyệt đối +20.3 **không** lớn hơn nhóm không nợ +21.4; cột '2.03x' chỉ là tỷ số rủi ro, lớn vì mức nền của họ thấp (19.7%). Các nhóm chưa có khoảng tin cậy nên em đọc thận trọng."

**📊 Đọc output:**
| Cột | Nghĩa |
| :--- | :--- |
| Risk(do(T=0)) / Risk(do(T=1)) | Tỷ lệ tốt nghiệp trong nhóm nếu không / có học bổng |
| Hiệu ứng CATE | Hiệu tuyệt đối (điểm %) — **dùng cột này để so nhóm** |
| Relative Risk "1.72x" | Tỷ số; **không** phải "hưởng lợi gấp 1.72 lần nhóm khác" |

### Cell 83 · code · 9.H Robustness Value — kết luận có dễ bị lật không

**🧩 Cell này làm gì:**
1. Hồi quy tốt nghiệp ~ học bổng + 5 biến nền; lấy hệ số và thống kê $t$ của học bổng ($t$ = 14.20, bậc tự do 3,623).
2. Tính $f = |t|/\sqrt{df}$ rồi **RV** theo công thức Cinelli & Hazlett (2020).
3. Bảng mốc: mỗi biến nền quan sát được giải thích bao nhiêu phần trăm phần dư của tốt nghiệp (partial $R^2$).

**🎤 Nói:**
"Câu hỏi: nếu có một biến ẩn em chưa đo — ví dụ động lực học tập — nó phải mạnh cỡ nào mới xóa được hiệu ứng? Đáp án: phải giải thích ít nhất **21%** phần dư của **cả** việc nhận học bổng lẫn việc tốt nghiệp. Để so sánh, biến quan sát mạnh nhất là học phí đúng hạn chỉ 10.4%, các biến khác dưới 5%. Vậy kết luận vững trước nhiễu ẩn yếu hơn học phí, nhưng không miễn nhiễm với một nhiễu mạnh gấp đôi học phí."

**📊 Đọc output:**
| Dòng | Nghĩa |
| :--- | :--- |
| Hệ số +0.2268, t = 14.20 | Học bổng tăng 22.7 điểm (hồi quy tuyến tính) |
| Partial $R^2$ 5.27% | Học bổng giải thích 5.3% phần dư của tốt nghiệp |
| $RV_{q=1}$ = 20.97% | Ngưỡng để **xóa** hiệu ứng |
| $RV_{q=1,\alpha=0.05}$ = 18.37% | Ngưỡng để hiệu ứng **mất ý nghĩa thống kê** |
| Bảng mốc | Tuổi 4.22% · điểm vào 0.48% · học vấn mẹ 0.03% · Debtor 0.81% · học phí 10.41% |


---

## MỤC 10 — KẾT LUẬN

### Cell 85 · markdown · 10.1 Trả lời RQ + 10.2 Khuyến nghị

**🎤 Nói:**
"Em quay lại bảng câu hỏi ở đầu bài.

* **RQ0:** dưới các giả định đã nêu, học bổng tăng xác suất tốt nghiệp **khoảng 24 điểm phần trăm**, ổn định qua cả ba mô hình và cả 5 seed.
* **RQ1:** đúng là chênh lệch thô bị thổi phồng. Khoảng 10 điểm, gần một phần ba, là do nhiễu.
* **RQ2 và RQ3:** phần học sâu **không** cải thiện dự báo một cách có ý nghĩa. H2 học được đồ thị hợp lệ và tự tìm ra cạnh $T \to Y$. Kết luận can thiệp không đổi. Em coi đây là một kết quả có giá trị: mô hình truyền thống B0 đã đủ tốt cho bài toán này.
* **RQ4:** kết luận khá vững. Refuter đạt, và RV bằng 21%, gấp đôi biến quan sát mạnh nhất. Nhưng nó không miễn nhiễm với nhiễu ẩn.
* **RQ5:** khoảng 21% tác động đi qua kết quả HK1. Nhóm lớn tuổi hưởng lợi nhiều hơn.

Hạn chế lớn nhất em tự nhận là biến học phí đúng hạn. Biến này có thể là hậu quả của việc bỏ học.

Bốn khuyến nghị: thí điểm phân bổ theo CATE, hỗ trợ học thuật sớm trong HK1 vì khoảng 21.5% tác động đi qua kỳ này, dùng refuter làm quy chuẩn kiểm định chính sách, và dùng mạng Bayes B0 làm lõi cho hệ thống cảnh báo sớm.

Em xin cảm ơn thầy và sẵn sàng nhận câu hỏi."

**⚠️ Lưu ý:** khuyến nghị 4 đã được viết lại cho khớp với null finding ở RQ2: B0 dự báo tương đương bản nơ-ron mà minh bạch hơn, nên B0 làm lõi cho hệ thống cảnh báo sớm. Nếu thầy hỏi "vậy làm H1/H2 để làm gì?", trả lời: để *kiểm chứng* rằng kết luận không phụ thuộc cách tham số hóa hay cách học đồ thị. Kết quả âm tính ấy chính là bằng chứng độ vững.
