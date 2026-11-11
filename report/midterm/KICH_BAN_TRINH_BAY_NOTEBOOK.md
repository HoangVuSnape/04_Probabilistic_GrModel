# KỊCH BẢN TRÌNH BÀY NOTEBOOK — TỪNG CELL MỘT

**Notebook:** `00_MASTER_Causal_PGM_Educational_Intervention.ipynb` (89 cell, chạy *Restart & Run All* ≈ 4–5 phút CPU)
**Dùng kèm:** `NOTEBOOK_DEFENSE_QA_SCRIPT.md` (các vòng chất vấn) · `glossary_de_hieu.md` (giải thích từ khóa)

## Cách dùng kịch bản này

* Mỗi cell có ba phần:
  * **🎤 Nói** — lời trình bày, đọc gần như nguyên văn được.
  * **📘 Từ khóa mới** — chỉ xuất hiện ở **lần đầu** từ khóa đó lộ ra trong notebook. Giải thích luôn tại chỗ, để thầy không phải hỏi lại. Lần sau gặp lại thì nói thẳng, không giải thích lại.
  * **⚠️ Lưu ý** — chỗ dễ bị hỏi vặn, hoặc chỗ notebook ghi chưa thật chuẩn mà mình phải nói cho đúng.
* Cell code: **không đọc code**. Nói hàm đó *làm gì*, rồi chỉ vào **output**.
* Nhịp gợi ý: mục 1–7 khoảng 10 phút, mục 8 khoảng 8 phút, mục 9–11 khoảng 7 phút.
* **Bẫy ký hiệu $Z_1..Z_5$:** thứ tự chỉ số **không giống nhau** giữa các phần (ghi chú ở cell 29). Khi trình bày, ưu tiên gọi **tên biến** ("điểm vào", "học phí đúng hạn") thay vì "Z3".

| Phần | $Z_1$ | $Z_2$ | $Z_3$ | $Z_4$ | $Z_5$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mục 6–7 (Domain DAG, DoWhy) | học vấn mẹ | điểm vào | Debtor | học phí đúng hạn | tuổi |
| **Mục 8 (B0/H1/H2, Tables A–C)** | **điểm vào** | **Debtor** | **học phí đúng hạn** | **học vấn mẹ** | **tuổi** |
| Mục 9.H (RV) và mục 10 | tuổi | điểm vào | học vấn mẹ | Debtor | học phí đúng hạn |

---

## MỤC 0 — MỞ ĐẦU

### Cell 0 · markdown · Tiêu đề và mục lục

**🎤 Nói:**
"Em xin trình bày đồ án môn Mô hình đồ thị xác suất. Câu hỏi của em rất đời thường: *nếu nhà trường cấp học bổng cho một sinh viên, khả năng sinh viên đó tốt nghiệp tăng thêm bao nhiêu?* Dữ liệu là bộ thật UCI 697, gồm hồ sơ sinh viên một trường đại học ở Bồ Đào Nha.

Toàn bộ đồ án nằm trong đúng một notebook này. Thầy bấm *Restart & Run All* thì mọi bảng, mọi hình và mọi đồ thị đều được sinh lại từ đầu, mất khoảng ba đến bốn phút trên CPU.

Về trình tự, em đi theo đúng mục lục. Mục 1 là dữ liệu và câu hỏi nghiên cứu. Mục 2 đến 7 là nền tảng lý thuyết. Mục 8 là phần chính: ba mô hình B0, H1, H2. Mục 9 và 10 là phần mở rộng. Mục 11 trả lời lại từng câu hỏi."

**📘 Từ khóa mới** *(mục lục nhắc rất nhiều từ; ở đây chỉ nói một câu cho mỗi từ, phần kỹ hơn nằm ở cell ghi trong ngoặc)*:
* **PGM — mô hình đồ thị xác suất:** biểu diễn một phân phối nhiều biến bằng đồ thị, trong đó nút là biến và cạnh là quan hệ phụ thuộc.
* **B0 / H1 / H2:** ba mô hình được so sánh. B0 là mạng Bayes truyền thống. H1 giữ đồ thị của B0 nhưng thay bảng xác suất của $Y$ bằng mạng nơ-ron. H2 dùng mạng nơ-ron để *học luôn đồ thị* (cell 37).
* **Tables A–C:** ba bảng kết quả bắt buộc. A là dự báo, B là can thiệp, C là độ vững và chi phí (cell 50, 54).
* **CATE:** hiệu ứng của học bổng *trong từng nhóm* sinh viên, ví dụ nhóm trên 20 tuổi (cell 78).
* **Trung gian (mediation):** tách xem học bổng tác động *trực tiếp*, hay *gián tiếp* qua kết quả học kỳ 1 (cell 76).

---

## MỤC 1 — GIỚI THIỆU, LÝ DO, CÂU HỎI, DỮ LIỆU

### Cell 1 · markdown · 1.0 Tuyên bố dùng AI + 1.1 Bối cảnh

**🎤 Nói:**
"Trước hết em minh bạch việc dùng AI. Em có dùng Claude Code và Gemini như một *lập trình viên đi cùng*: chuẩn hóa cú pháp, vẽ hình, viết docstring tiếng Việt. Còn các quyết định phương pháp luận do em tự đưa ra và chịu trách nhiệm. Đó là thứ tự thời gian của các biến, danh sách cạnh bị cấm, tham số làm mịn, bộ kiểm tra DAG, và nguyên tắc *không đụng tới nhãn của tập test* khi ước lượng can thiệp."

**📘 Từ khóa mới:**
* **Tầng thời gian $\mathbf{Z} \to T \to M_1 \to Y$:** các biến được xếp theo thứ tự xảy ra. Biến nền có trước, rồi tới học bổng, rồi tới kết quả học kỳ 1, cuối cùng là tốt nghiệp. Nguyên tắc: cái xảy ra sau không thể là nguyên nhân của cái xảy ra trước.
* **Ma trận mặt nạ $M$:** một bảng 0/1 đánh dấu cạnh nào *bị cấm*. Đem nhân vào trọng số của đồ thị thì các cạnh bị cấm luôn bằng 0.
* **NOTEARS, $h(W)=0$:** một cách học đồ thị bằng tối ưu liên tục. $h(W)$ là một hàm bằng 0 *khi và chỉ khi* đồ thị không có chu trình (cell 48).
* **BDeu, ESS:** cách làm mịn bảng xác suất bằng cách cộng thêm "mẫu ảo". ESS là số mẫu ảo đó (cell 24–26).
* **$C_{\text{test}}$ ($n = 545$):** tập hồ sơ sinh viên của phần test, *chỉ lấy các biến nền*, dùng làm "quần thể chuẩn" để tính hiệu ứng.
* **$Y_{\text{test}}$:** nhãn tốt nghiệp thật của tập test. Chỉ dùng để chấm dự báo, **không** dùng để tính hiệu ứng can thiệp.

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
"Câu hỏi chính RQ0 được viết thành một đại lượng cụ thể, gọi là ATE. Ngoài ra có sáu câu phụ:
* RQ1: chênh lệch thô có bị thổi phồng không?
* RQ2 và RQ3: phần học sâu, H1 và H2, có giúp gì không?
* RQ4: kết luận vững tới đâu?
* RQ5 và RQ6 là phần mở rộng: cơ chế tác động, và cách phân bổ học bổng.

Cột cuối của bảng ghi rõ chỗ trả lời từng câu. Ở mục 11, em quay lại bảng này và trả lời từng dòng."

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

### Cell 5 · markdown · 1.0b Cài đặt và tái lập

**🎤 Nói:**
"Phần này hướng dẫn chạy lại trên máy cá nhân hoặc trên Colab. Phiên bản thư viện được ghim trong `requirements.txt`. Dữ liệu và các file split đã có sẵn trong repo, nên không cần tải thêm."

**📘 Từ khóa mới:**
* **Split:** file JSON ghi lại *chính xác* dòng nào thuộc TRAIN, dòng nào thuộc VAL, dòng nào thuộc TEST. Đóng băng như vậy để chạy lại lúc nào cũng ra cùng kết quả.

### Cell 6 · code · In môi trường

**🎤 Nói:**
"Cell này in đúng môi trường của lần chạy này: Python 3.13, pgmpy 1.1.2, DoWhy 0.14, PyTorch 2.15 chạy CPU, cùng với revision git. Ai muốn tái lập kết quả thì biết chính xác cần dựng môi trường nào."

### Cell 7 · markdown · 1.4 Đọc kết quả nạp dữ liệu

**🎤 Nói:**
"Hai nhóm có học bổng và không có học bổng đều đủ lớn khi nhìn tổng thể. Nhưng em lưu ý một điều: đủ lớn tổng thể *chưa* có nghĩa là đủ trong từng nhóm con. Điều kiện cần thật sự là positivity, và em sẽ kiểm tra theo từng ô ở mục 1.6 và mục 8.2b."

**📘 Từ khóa mới:**
* **Positivity / overlap (điều kiện dương):** với mọi loại hồ sơ $\mathbf{z}$, phải có *cả* người nhận lẫn người không nhận học bổng, tức là $0 < P(T=1 \mid \mathbf{Z}=\mathbf{z}) < 1$. Nếu một nhóm toàn người có học bổng thì không có ai để so sánh, và mô hình buộc phải *đoán mò* (ngoại suy).

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

### Cell 10 · markdown · 1.6 Kiểm toán bổ sung trước khi mô hình hóa

**🎤 Nói:**
"Trước khi dựng mô hình, em làm bốn kiểm tra để trả lời bốn câu hỏi thiết kế:
* Biến nào được phép làm $\mathbf{Z}$?
* Ngưỡng chia điểm đầu vào có hợp lý không?
* Biến nào liên quan mạnh tới kết cục?
* Mỗi nhóm hồ sơ có đủ cả người có và không có học bổng không?"

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

### Cell 15 · markdown · 1.6c Đọc kết quả kiểm toán

**🎤 Nói:**
"Em tóm lại bốn điều rút ra từ phần kiểm toán:
* Các biến sau nhập học đã bị loại khỏi tập điều chỉnh.
* Ngưỡng điểm hợp lý, và điểm vào đúng là một biến nhiễu.
* Biến học phí cần thận trọng.
* Overlap thô ổn. Khi chia mịn đủ cả 5 biến, vẫn còn vài hồ sơ thiếu hỗ trợ. Những hồ sơ này được ghi ở cột `Overlap_Status` của Table B."

---

## MỤC 2 — CAN THIỆP VÀ CHÊNH LỆCH THÔ

### Cell 16 · markdown · 2.1–2.2 Nghịch lý quan sát, cơ chế chính sách

**🎤 Nói:**
"Đây là ý tưởng cốt lõi của cả đồ án. Xác suất tốt nghiệp *khi quan sát thấy* một sinh viên có học bổng, $P(Y \mid T=1)$, **khác** với xác suất tốt nghiệp *khi ta ép* cấp học bổng, $P(Y \mid do(T=1))$.

Lý do là người được nhận học bổng vốn đã khác người không được nhận: điểm cao hơn, trẻ hơn, tài chính ổn hơn. Phần khác biệt sẵn có này *trộn lẫn* với hiệu ứng thật của học bổng.

Em cũng mô tả cụ thể cơ chế chính sách:
* Can thiệp xảy ra vào tuần định hướng, trước bất kỳ buổi học nào.
* Thông tin có trước can thiệp chỉ gồm 5 biến nền.
* Tín chỉ HK1 xảy ra sau, nên tuyệt đối không được dùng để điều chỉnh."

**📘 Từ khóa mới:**
* **$P(Y \mid T)$ so với $P(Y \mid do(T))$:** vế đầu là *nhìn* (lọc ra những người có $T=1$ trong dữ liệu). Vế sau là *làm* (can thiệp, ép mọi người $T=1$). Hai đại lượng chỉ bằng nhau khi không có nhiễu, ví dụ trong thí nghiệm ngẫu nhiên.
* **Confounding bias (thiên lệch do nhiễu):** phần chênh lệch đến từ việc hai nhóm vốn khác nhau, chứ không phải do học bổng.
* **Non-manipulability:** tuổi, giới tính, học vấn của mẹ là những thứ nhà trường không can thiệp được. Vì vậy chúng chỉ đóng vai trò biến nền, không thể là biến can thiệp.
* **Mediator / collider bias:** điều chỉnh theo biến trung gian sẽ chặn mất một phần hiệu ứng thật. Điều chỉnh theo một *collider* (biến bị nhiều mũi tên trỏ vào) sẽ mở ra một tương quan giả.

### Cell 17 · code · Tính chênh lệch thô

**🎤 Nói:**
"Đây là phép tính ngây thơ nhất. Sinh viên có học bổng tốt nghiệp 86.17%, sinh viên không có học bổng tốt nghiệp 51.63%. Chênh lệch thô là **+34.54 điểm phần trăm**. Con số này là điểm xuất phát. Cả đồ án là để bóc xem trong 34.54 điểm đó, bao nhiêu là nhân quả thật."

**📘 Từ khóa mới:**
* **Naive gap (chênh lệch thô):** $P(Y=1 \mid T=1) - P(Y=1 \mid T=0)$. Đây chỉ là liên kết, chưa điều chỉnh gì.

### Cell 18 · code · Hình 2.1

**🎤 Nói:**
"Hình này vẽ lại hai tỷ lệ vừa tính để thấy trực quan khoảng cách 34.54 điểm."

### Cell 19 · markdown · 2.3 Đọc kết quả

**🎤 Nói:**
"Nếu nhà trường tin rằng cứ cấp học bổng là tăng 34.5 điểm tốt nghiệp, họ sẽ đánh giá quá cao hiệu quả ngân sách. Phần sau sẽ cho thấy, sau khi điều chỉnh nhiễu, con số chỉ còn khoảng 24 điểm. Tức khoảng 10 điểm, gần một phần ba, là do hai nhóm vốn khác nhau."

---

## MỤC 3 — HỌC CẤU TRÚC (MINH HỌA)

### Cell 20 · markdown · 3.1–3.2 Ba họ thuật toán học cấu trúc

**🎤 Nói:**
"Khi chưa biết đồ thị, ta có thể học nó từ dữ liệu. Các thư viện hiện nay chia thành ba họ, khác nhau ở *câu hỏi* mà thuật toán đặt ra cho dữ liệu.

**Họ thứ nhất: dựa trên ràng buộc độc lập.** Câu hỏi là: *hai biến này có độc lập với nhau không, sau khi đã biết một số biến khác?*
* **PC** bắt đầu từ đồ thị nối tất cả các cặp biến. Với mỗi cặp, nó làm kiểm định thống kê: nếu hai biến độc lập khi đã biết một nhóm biến nào đó thì xóa cạnh giữa chúng. Sau khi tỉa xong, nó định chiều các cạnh. Trước hết là các v-structure $X \to Z \leftarrow Y$, vì đây là dạng duy nhất dữ liệu cho phép xác định chiều. Rồi nó dùng quy tắc Meek để suy ra chiều của những cạnh còn lại mà không tạo chu trình hay v-structure mới.
* **FCI** là PC mở rộng cho trường hợp có biến nhiễu ẩn. Vì không dám chắc chiều cạnh nào, kết quả của nó là một PAG: một số đầu cạnh được đánh dấu 'không xác định'.

**Họ thứ hai: dựa trên hàm điểm.** Câu hỏi là: *đồ thị nào giải thích dữ liệu tốt nhất?* Mỗi đồ thị được chấm một điểm, như BIC hoặc BDeu, rồi thuật toán tìm đồ thị có điểm cao nhất.
* **HillClimb (leo đồi)** xuất phát từ một đồ thị, thử từng thay đổi nhỏ (thêm, xóa hoặc đảo một cạnh), giữ lại thay đổi làm điểm tăng nhiều nhất, và lặp lại cho tới khi không cải thiện được nữa. Điểm mạnh của nó là nhận `ExpertKnowledge` rất dễ: cạnh nào bị cấm thì không bao giờ được thử.
* **GES** cũng tìm kiếm tham lam, nhưng trên không gian các *lớp tương đương* thay vì từng DAG riêng lẻ. Về lý thuyết, nếu dữ liệu đủ lớn, nó sẽ tìm ra đúng lớp tương đương.

**Họ thứ ba: tối ưu liên tục.** Đại diện là **NOTEARS**. Thay vì thử từng đồ thị, nó biểu diễn đồ thị bằng một ma trận trọng số $W$ rồi dùng gradient để tối ưu, giống cách huấn luyện mạng nơ-ron. Khó khăn là bảo đảm kết quả không có chu trình. NOTEARS giải quyết bằng hàm $h(W)$: hàm này bằng 0 khi và chỉ khi đồ thị không có chu trình, nên chỉ cần ép $h(W) = 0$ trong lúc tối ưu.

Bảng 3.2 tóm tắt ưu nhược điểm của từng họ. Em chọn **HillClimb cho B0**, vì dữ liệu đã rời rạc hóa và cần áp ràng buộc thời gian rất chặt. Em chọn **NOTEARS-MLP cho H2**, vì đây là cách học cấu trúc mà có thể dùng mạng nơ-ron, và ràng buộc thời gian đưa vào được qua mặt nạ $M$."

**📘 Từ khóa mới:**
* **DAG — Directed Acyclic Graph:** đồ thị có hướng, không có chu trình, tức không có vòng $A \to B \to A$. Đây là "khung xương" của mạng Bayes.
* **Kiểm định độc lập có điều kiện:** kiểm tra xem $X$ và $Y$ có còn liên quan với nhau không khi đã cố định $Z$. Ví dụ: điểm vào và tốt nghiệp có thể liên quan, nhưng nếu điều đó hoàn toàn đi qua học bổng thì khi đã biết học bổng, hai biến sẽ độc lập. **$\chi^2$** là kiểm định dùng cho biến rời rạc (so số đếm thực tế với số đếm kỳ vọng nếu độc lập). **Fisher-Z** dùng cho biến liên tục (kiểm tra hệ số tương quan riêng có bằng 0 không).
* **v-structure (collider):** cấu trúc $X \to Z \leftarrow Y$. $X$ và $Y$ độc lập với nhau, nhưng khi biết $Z$ thì chúng trở nên liên quan. Dấu hiệu thống kê đặc biệt này giúp xác định được chiều cạnh từ dữ liệu.
* **Quy tắc Meek:** bộ quy tắc suy luận chiều cho các cạnh còn lại sau khi đã định chiều các v-structure, sao cho không sinh chu trình và không sinh v-structure mới.
* **Lớp tương đương Markov / CPDAG:** nhiều DAG khác nhau có thể sinh ra *cùng* tập quan hệ độc lập, ví dụ $A \to B \to C$ và $A \leftarrow B \leftarrow C$. Dữ liệu quan sát không phân biệt được chúng. CPDAG là đồ thị đại diện cho cả lớp: cạnh nào chắc chiều thì có mũi tên, cạnh nào không chắc thì để vô hướng.
* **PAG (Partial Ancestral Graph):** kết quả của FCI. Giống CPDAG nhưng cho phép thêm ký hiệu "có thể có biến ẩn ở đây".
* **Cực trị địa phương:** HillClimb chỉ đi lên. Nếu đứng ở một "đỉnh đồi nhỏ" thì mọi bước đi đều làm điểm giảm, nên nó dừng lại dù ở chỗ khác có "đỉnh núi" cao hơn.
* **NP-hard:** số DAG có thể có tăng siêu lũy thừa theo số biến. Chỉ với 7 biến đã có khoảng $10^9$ DAG, nên không thể thử hết.
* **$h(W) = \operatorname{tr}(e^{W \odot W}) - d$:** $W \odot W$ bình phương từng phần tử để mọi trọng số đều dương. $e^{(\cdot)}$ cộng dồn tất cả các đường đi vòng. Vết $\operatorname{tr}$ đếm các đường đi quay về chính nó. Nếu không có chu trình, vết đúng bằng $d$ (số biến), nên $h(W) = 0$.
* **"Dễ bị đảo chiều do lệch phương sai":** NOTEARS bản tuyến tính có xu hướng đặt biến có phương sai nhỏ làm nguyên nhân, biến có phương sai lớn làm kết quả. Vì vậy chỉ cần đổi đơn vị đo cũng có thể làm đảo chiều cạnh. Đây là lý do H2 cần mặt nạ thời gian $M$ để chặn các cạnh ngược.

**⚠️ Lưu ý:**
* Nếu thầy hỏi *"sao không dùng PC?"*: PC phải chạy rất nhiều kiểm định. Với các ô dữ liệu thưa thì kiểm định yếu, và sai sót tích lũy dần (xóa nhầm một cạnh ở bước đầu thì các bước sau định chiều sai theo). Ngoài ra, kết quả của PC là một CPDAG có thể còn cạnh vô hướng, trong khi mạng Bayes cần một DAG đầy đủ chiều để tính $do()$.
* Nếu thầy hỏi *"sao không dùng FCI, vì nó xử lý được nhiễu ẩn?"*: FCI cho ra PAG, không đủ để tính CPD và ATE. Nhóm chọn cách khác: giả định không có nhiễu ẩn, rồi đo xem kết luận nhạy tới đâu bằng RV ở mục 9.H.
* Không nói HillClimb "tìm ra đồ thị đúng". Nó chỉ tìm đồ thị có điểm cao, và đồ thị đó có thể dừng ở cực trị địa phương. Vì vậy B0 chạy 5 seed để kiểm tra độ ổn định.

### Cell 21 · code · HillClimb có và không có ràng buộc

**🎤 Nói:**
"Cell này minh họa vì sao học cấu trúc *bắt buộc* phải có ràng buộc thời gian. Em chạy HillClimb với điểm BIC trên 4 biến, hai lần.

Lần thứ nhất em để thuật toán *học tự do*. Nó học ra những cạnh vô lý như `Tốt nghiệp → Điểm đầu vào`, tức kết cục cuối khóa lại gây ra điểm thi trước khi nhập học. Lý do là dữ liệu quan sát chỉ cho biết hai biến có liên quan, chứ không tự cho biết chiều.

Lần thứ hai em thêm `ExpertKnowledge` để cấm mọi cạnh đi ra từ $Y$. Những cạnh ngược đó biến mất. Nhưng như thầy sẽ thấy ở hình ngay sau, vẫn còn một cạnh sai."

**📘 Từ khóa mới:**
* **BIC — Bayesian Information Criterion:** điểm bằng log-likelihood trừ đi một mức phạt theo số tham số. Nó khuyến khích đồ thị vừa khớp dữ liệu vừa đơn giản.
* **`ExpertKnowledge`:** cách khai báo cho pgmpy danh sách cạnh *cấm* và cạnh *bắt buộc* trước khi học.

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

## MỤC 4 — HỌC THAM SỐ

### Cell 24 · markdown · 4.1–4.2 CPD và bảng so sánh

**🎤 Nói:**
"Có đồ thị rồi, bước tiếp theo là học các bảng xác suất. Mỗi nút cần một phân phối *có điều kiện theo cha của nó*. Bảng này so sánh bốn cách học: MLE, BDeu, EM và Neural CPD."

**📘 Từ khóa mới:**
* **CPD — Conditional Probability Distribution:** $P(X_i \mid \text{Pa}(X_i))$, xác suất của một biến khi biết các biến cha của nó. Với biến rời rạc, CPD là một bảng, gọi là CPT.
* **$\text{Pa}(X)$ — tập cha:** các nút có mũi tên đi thẳng vào $X$.
* **Phân rã của mạng Bayes:** $P(X_1..X_n) = \prod_i P(X_i \mid \text{Pa}(X_i))$. Phân phối chung bằng tích các CPD.
* **MLE — Maximum Likelihood:** đếm tần suất, $\hat\theta = N_{ijk}/N_{ij}$. Nếu một ô không có mẫu nào thì xác suất bằng 0, hoặc không định nghĩa được.
* **BDeu:** cộng thêm mẫu ảo theo tiền nghiệm Dirichlet, $\hat\theta = (N_{ijk} + \alpha_{ijk})/(N_{ij} + \alpha_{ij})$, nên không có ô nào ra 0 tuyệt đối.
* **EM:** thuật toán lặp, dùng khi dữ liệu bị thiếu.
* **Neural CPD:** thay bảng bằng một mạng nơ-ron, nhận vào giá trị của các biến cha và trả ra xác suất.

### Cell 25 · code · MLE và BDeu trên cùng một ô

**🎤 Nói:**
"Em đặt hai bảng CPD của $Y$ cạnh nhau, với cha là Debtor và học bổng. Ô *đang nợ học phí và có học bổng* có rất ít sinh viên. MLE cho xác suất tốt nghiệp khoảng 0.597. BDeu với ESS bằng 10 kéo con số này về gần mức chung hơn. Đó chính là tác dụng làm mịn."

### Cell 26 · markdown · 4.4 Đọc kết quả

**🎤 Nói:**
"Ý nghĩa thực tế: trong một hệ thống cảnh báo sớm, BDeu giúp mô hình không bao giờ nói *xác suất 0%* cho một tình huống chỉ đơn giản là hiếm gặp.

Giới hạn: khi có nhiều cha, bảng sẽ nổ ra rất nhiều ô. Ví dụ 10 cha nhị phân là 1,024 cột. Đó là lý do em thử H1 với CPD nơ-ron."

---

## MỤC 5 — BỘ KIỂM TRA DAG

### Cell 27 · markdown · 5.1 Bốn ràng buộc

**🎤 Nói:**
"Trước khi tin một đồ thị, em kiểm tra bốn điều:
* Không có chu trình.
* Không có cạnh đi ngược thời gian.
* Đúng danh sách cấm và bắt buộc của chuyên gia.
* Tập $\mathbf{Z}$ chặn hết mọi đường cửa sau từ $T$ tới $Y$."

**📘 Từ khóa mới:**
* **Đường cửa sau (backdoor path):** đường nối $T$ với $Y$ mà *bắt đầu bằng một mũi tên đi vào $T$*, ví dụ $T \leftarrow Z \rightarrow Y$. Đường này chở tương quan giả từ $Z$.
* **d-separation:** tiêu chí đồ thị để biết hai biến có độc lập khi biết một tập biến khác hay không. Một đường bị "chặn" nếu ta điều kiện trên một nút không phải collider nằm trên đường đó.
* **Tiêu chuẩn cửa sau:** tập $\mathbf{Z}$ hợp lệ nếu (1) không chứa hậu duệ của $T$, và (2) chặn mọi đường cửa sau. Khi đó:
  $$P(y \mid do(t)) = \sum_{\mathbf{z}} P(y \mid t, \mathbf{z})\,P(\mathbf{z})$$
  Đây là **công thức điều chỉnh cửa sau** (backdoor adjustment), dùng xuyên suốt đồ án.
* **$\mathcal{G}_{\underline{T}}$:** đồ thị sau khi xóa mọi cạnh *đi ra* từ $T$. Trên đồ thị này, những đường còn nối $T$ với $Y$ chính là các đường cửa sau.

### Cell 28 · code · Bốn hàm kiểm tra

**🎤 Nói:**
"Cell này chỉ định nghĩa bốn hàm kiểm tra. Mỗi hàm trả về một kết quả có cấu trúc, gồm đạt hay không đạt và lý do. Cell tiếp theo sẽ dùng chúng."

---

## MỤC 6 — DOMAIN DAG VÀ GRAPH SURGERY

### Cell 29 · markdown · 6.1 Domain DAG

**🎤 Nói:**
"Đây là đồ thị do em xây theo kiến thức miền. Bản 7 nút có 11 cạnh: mỗi biến nền trỏ vào cả $T$ và $Y$, cộng thêm cạnh $T \to Y$. Bản 8 nút thêm biến trung gian $M_1$ là tín chỉ HK1, nên có 13 cạnh.

Em xin lưu ý thầy phần ghi chú cuối cell: thứ tự chỉ số $Z_1$ đến $Z_5$ **không** giống nhau giữa các phần. Vì vậy khi trình bày em sẽ gọi bằng tên biến."

**📘 Từ khóa mới:**
* **Domain DAG:** đồ thị do chuyên gia vẽ, không học từ dữ liệu. Trong đồ án, nó là *tham chiếu*, để so với đồ thị học được ở mục 8.

### Cell 30 · code · Chạy bốn kiểm tra trên Domain DAG

**🎤 Nói:**
"Cả bốn kiểm tra đều đạt: không có chu trình, không có cạnh ngược thời gian, đúng ràng buộc chuyên gia, và 5 biến nền chặn hết đường cửa sau."

### Cell 31 · code · Hình 6.1

**🎤 Nói:**
"Hình 6.1 có hai panel. Cột bên trái là 5 biến nền, *Tốt nghiệp* nằm bên phải. Học bổng và tín chỉ HK1 được đặt ở hàng dưới, để thầy thấy rõ hai loại đường đi: các mũi tên ngang đi thẳng từ biến nền sang kết cục, và các mũi tên chéo xuống đi vào học bổng.

Panel (a) là đồ thị gốc, với các mũi tên từ biến nền đi vào học bổng. Panel (b) là đồ thị *sau phẫu thuật*: mọi mũi tên đi vào $T$ đã bị cắt (nét đứt màu đỏ), còn các mũi tên đi thẳng từ biến nền sang kết cục vẫn giữ nguyên."

**📘 Từ khóa mới:**
* **Graph surgery (phẫu thuật đồ thị):** cách Pearl mô tả phép $do(T=t)$. Ta cắt mọi cạnh *đi vào* $T$ rồi gán cố định $T=t$. Ý nghĩa: khi nhà trường *quyết định* cấp học bổng, học bổng không còn phụ thuộc vào điểm hay tuổi của sinh viên nữa.

### Cell 32 · markdown · 6.1b Đọc Hình 6.1

**🎤 Nói:**
"Ở panel (a), các đường $T \leftarrow Z \rightarrow Y$ chính là nguồn gốc của 34.54 điểm chênh lệch thô. Ở panel (b), sau khi cắt, $T$ chỉ còn tác động lên $Y$ qua các đường nhân quả: trực tiếp, hoặc đi qua $M_1$.

Điều chỉnh theo $\mathbf{Z}$ cho phép tái tạo đúng phân phối sau phẫu thuật từ dữ liệu quan sát. Còn $M_1$ là biến hậu can thiệp nên không được điều chỉnh."

**📘 Từ khóa mới:**
* **Truncated factorization (phân rã cắt cụt):**
  $$P(\mathbf{z}, m, y \mid do(T=t)) = \prod_k P(z_k)\,P(m \mid t)\,P(y \mid t, m, \mathbf{z})$$
  So với phân rã gốc, ta chỉ bỏ đi đúng một thừa số $P(t \mid \mathbf{z})$.

### Cell 33 · markdown · 6.2 Đọc kết quả

**🎤 Nói:**
"Kết luận của mục 6: đồ thị hợp lệ, nên hiệu ứng *định danh được*. Nghĩa là ATE viết được thành một công thức chỉ chứa phân phối quan sát.

Em xin nói cho chính xác: ước lượng là *không chệch* với điều kiện **các giả định đúng**, đặc biệt là giả định không có nhiễu ẩn. Định danh được không có nghĩa là đã chứng minh được kết luận."

**📘 Từ khóa mới:**
* **Identifiability (tính định danh):** hiệu ứng can thiệp tính được từ dữ liệu quan sát *nếu* đồ thị đúng. Đây là câu hỏi về toán, không phải về cỡ mẫu.

**⚠️ Lưu ý:** cell này ghi "100% đạt… ước lượng không chệch". Khi nói, luôn kèm vế "*dưới giả định đồ thị đúng và không có nhiễu ẩn*".

---

## MỤC 7 — DOWHY

### Cell 34 · markdown · 7.1 Bốn bước của DoWhy

**🎤 Nói:**
"DoWhy của Microsoft chia một phân tích nhân quả thành bốn bước. **Model**: đưa đồ thị vào. **Identify**: tìm tập điều chỉnh. **Estimate**: ước lượng ATE. **Refute**: cố tình phá kết quả để xem nó có đứng vững không."

**📘 Từ khóa mới:**
* **DoWhy:** thư viện Python cho suy luận nhân quả, và là thư viện bắt buộc của đề bài.
* **Linear Regression (backdoor):** hồi quy $Y$ theo $T$ và $\mathbf{Z}$. Hệ số của $T$ chính là ATE.
* **Propensity Score Weighting:** ước lượng xác suất nhận học bổng $e(\mathbf{z}) = P(T=1 \mid \mathbf{z})$, gọi là propensity score. Sau đó đánh trọng số mỗi sinh viên bằng $1/e$ hoặc $1/(1-e)$ để hai nhóm trở nên "giống nhau" về $\mathbf{Z}$.
* **Refuter:** phép thử bác bỏ. Nếu kết quả là thật thì nó phải sống sót qua các phép thử này.

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

### Cell 37 · markdown · 8.1 Thiết kế B0/H1/H2 + 8.2 Multi-treatment

**🎤 Nói:**
"Đây là phần chính của đồ án. Cả ba mô hình dùng *cùng 7 biến*, *cùng các dòng TRAIN, VAL, TEST*, và *cùng ràng buộc thời gian*:
* **B0** học đồ thị bằng HillClimb, rồi học CPD bằng BDeu.
* **H1** giữ nguyên đồ thị của B0, chỉ thay CPD của $Y$ bằng một mạng MLP.
* **H2** học đồ thị bằng NOTEARS-MLP, với cùng mặt nạ ràng buộc.

Bảng H3 chỉ là phụ lục, sinh từ script riêng, và không so sánh trực tiếp với Tables A–C.

Mục 8.2 thử thêm hai can thiệp khác. $T_2$ là trạng thái học phí đúng hạn. $T_3$ là bật cả hai cùng lúc. Em nói ngay từ đầu: hai can thiệp này **chỉ mang tính khám phá**, vì $T_2$ có nguy cơ nhân quả ngược và $T_3$ thiếu positivity."

**📘 Từ khóa mới:**
* **MLP — Multi-Layer Perceptron:** mạng nơ-ron truyền thẳng nhiều lớp.
* **Checkpoint `.pt`:** file lưu trọng số mạng PyTorch sau khi huấn luyện, có thể nạp lại mà không cần huấn luyện lại.
* **Mặt nạ $M$ và $h(W) < 10^{-6}$:** đã nói ở cell 1. Ở đây $10^{-6}$ là ngưỡng số học để coi như "đã hết chu trình".
* **Multi-treatment:** xét nhiều biến can thiệp. $T_3$ là *tương phản kép*: $do(T_1=1, T_2=1)$ so với $do(T_1=0, T_2=0)$.
* **Nhân quả ngược (reverse causation):** chiều thật là $Y \to T$ chứ không phải $T \to Y$. Ví dụ ở đây: bỏ học dẫn tới ngừng đóng phí.
* **Outcome Regression (T-learner):** khớp riêng hai mô hình kết cục, một cho $T=1$ và một cho $T=0$, rồi lấy hiệu dự đoán của hai mô hình.
* **AIPW:** giải thích ở cell 74.

### Cell 38 · markdown · 8.0 Pipeline chạy trực tiếp

**🎤 Nói:**
"Từ đây, các cell 8.A đến 8.F là *toàn bộ* code sinh ra Tables A–C. Không có gì chạy ngầm ở chỗ khác. Lưu ý là mã hóa 7 biến ở phần này dùng thứ tự riêng: $Z_1$ là điểm vào, $Z_2$ là Debtor, $Z_3$ là học phí, $Z_4$ là học vấn của mẹ, $Z_5$ là tuổi."

### Cell 39 · code · Nạp thư viện cho pipeline

**🎤 Nói:**
"Cell này nạp thư viện và in phiên bản. PyTorch chạy trên CPU, không có CUDA. Như vậy kết quả không phụ thuộc GPU, và thầy chạy trên laptop cũng được."

### Cell 40 · markdown · 8.A Cấu hình chung và dữ liệu 7 biến

**🎤 Nói:**
"Mọi thí nghiệm dùng chung *một* đối tượng `Config`. Dữ liệu được chia 70/15/15, *phân tầng theo cả $Y$ và $T$*, với 5 seed từ 42 tới 46. Tiền xử lý chỉ được fit trên TRAIN của từng seed."

**📘 Từ khóa mới:**
* **Stratified split (chia phân tầng):** chia sao cho tỷ lệ tốt nghiệp và tỷ lệ học bổng trong mỗi phần giống với toàn bộ dữ liệu.
* **Seed:** hạt giống ngẫu nhiên. Cố định seed thì chạy lại sẽ ra cùng kết quả. Dùng 5 seed để xem kết quả dao động bao nhiêu.

**⚠️ Lưu ý:** cell này còn sót một dòng tiêu đề cũ "7.1. Đặc tả hợp đồng…". Đó chỉ là tiêu đề, nội dung đúng. Nếu thầy hỏi, nói đây là tiêu đề còn lại từ notebook trước khi gộp.

### Cell 41 · code · Config, thư mục, seed

**🎤 Nói:**
"Cell này khởi tạo cấu hình, tạo các thư mục `artifacts/`, và cài hàm cố định seed cho numpy, random và torch."

### Cell 42 · code · Nạp và mã hóa 7 biến

**🎤 Nói:**
"Từ 4,424 dòng, sau khi loại 794 Enrolled còn 3,630 dòng, mã hóa thành 7 cột. Cân bằng nhãn: 2,209 tốt nghiệp và 1,421 bỏ học. Cân bằng can thiệp: 969 có học bổng và 2,661 không có. Các con số khớp với mục 1."

### Cell 43 · code · Tạo hoặc đọc 5 split

**🎤 Nói:**
"Cả 5 seed đều chia thành 2,541 TRAIN, 544 VAL và 545 TEST. Split được lưu ra file JSON và đóng băng. B0, H1, H2 đọc *chính xác cùng các dòng*, nên đây là so sánh **ghép cặp**."

**📘 Từ khóa mới:**
* **Paired run (run ghép cặp):** ba mô hình chạy trên cùng một split ở mỗi seed. Nhờ đó, khác biệt giữa các mô hình không bị lẫn với khác biệt do cách chia dữ liệu.

### Cell 44 · markdown · 8.B Thiết kế B0

**🎤 Nói:**
"B0 được học theo đúng trình tự sau:
1. Khai báo ràng buộc **trước** khi học: cấm $Y$ trỏ ngược về $T$ hay $Z$, cấm $T$ trỏ về $Z$, và **bắt buộc** cạnh $T \to Y$.
2. Chạy HillClimb với ba hàm điểm.
3. Chọn hàm điểm theo log-likelihood trên VALIDATION.
4. Đóng băng đồ thị trước khi đụng tới TEST.

Hiệu ứng được tính bằng cách chuẩn hóa trên $C_{\text{test}}$."

**📘 Từ khóa mới:**
* **Vì sao bắt buộc $T \to Y$:** nếu thuật toán không học ra cạnh này thì ATE bằng 0 *theo định nghĩa*, chứ không phải do dữ liệu nói vậy. $T \to Y$ là *giả thuyết cần đo*, nên phải giữ để đo nó.
* **Log-likelihood trên VALIDATION:** mô hình học trên TRAIN gán xác suất cao tới đâu cho dữ liệu mà nó chưa từng thấy. Càng cao thì mô hình càng tốt.
* **K2:** một hàm điểm Bayes, tương tự BDeu nhưng với tiền nghiệm đều.
* **$q_m(t, C_i)$:** xác suất tốt nghiệp mà mô hình $m$ dự đoán cho hồ sơ $i$ *nếu ép* $T=t$:
  $$q_m(t, C_i) = P_m(Y=1 \mid T=t,\ \text{Pa}(Y)\setminus T = C_i)$$
* **risk và ATE chuẩn hóa:**
  $$\text{risk}_m(t) = \tfrac{1}{n}\sum_i q_m(t, C_i), \qquad \text{ATE}_m = \text{risk}_m(1) - \text{risk}_m(0)$$
  Tức là lấy 545 hồ sơ test, *giả sử* tất cả đều có học bổng rồi tính trung bình, sau đó giả sử tất cả đều không có rồi tính trung bình, và lấy hiệu. Đây chính là công thức cửa sau, chỉ khác là $P(\mathbf{z})$ được thay bằng phân bố thực tế của tập test. **Không dùng $Y_{\text{test}}$.**
* **Vì sao $\text{Pa}(Y)\setminus T$ là tập điều chỉnh hợp lệ:** vì mặt nạ cấm $T \to Z$, nên mọi cha khác của $Y$ đều là biến có trước can thiệp. Chúng chặn được đường cửa sau và không phải hậu duệ của $T$.

### Cell 45 · code · Huấn luyện B0 (seed 42)

**🎤 Nói:**
"Ở seed 42, ba hàm điểm cho ba đồ thị. BIC và BDeu cùng ra 9 cạnh. K2 ra 11 cạnh, với log-likelihood trên VAL cao nhất (−2208.37 so với −2211.11), nên **K2 được chọn**. Thực tế K2 được chọn ở cả 5 seed.

Cha của $Y$ là học bổng, Debtor, học phí và tuổi. Điểm vào và học vấn của mẹ không được nối thẳng vào $Y$.

Hiệu ứng trên tập test: risk khi không cấp là 0.540, risk khi cấp là 0.786, ATE là **+0.2467**."

### Cell 46 · markdown · 8.C Thiết kế H1

**🎤 Nói:**
"H1 sao chép đồ thị B0 của *đúng run đó*, và chỉ thay CPD của $Y$. Đầu vào của MLP là đúng tập cha của $Y$. Mạng được dừng sớm theo VALIDATION. Sau đó em liệt kê mọi tổ hợp giá trị của các cha, chạy qua MLP để đổ thành một `TabularCPD`. Nhờ vậy H1 vẫn là một mạng Bayes hợp lệ trong pgmpy."

**📘 Từ khóa mới:**
* **BCEWithLogitsLoss:** hàm mất mát entropy chéo nhị phân, dùng để huấn luyện mô hình xác suất 0/1.
* **Early stopping (dừng sớm):** ngừng huấn luyện khi loss trên VALIDATION không giảm sau 15 epoch liên tiếp (patience), để tránh học vẹt.
* **BatchNorm, Dropout:** hai kỹ thuật giúp mạng ổn định và bớt overfit.
* **Clip xác suất $[10^{-4}, 1-10^{-4}]$:** chặn số học để tránh $\log 0$. Việc này tách biệt với làm mịn BDeu.

### Cell 47 · code · Huấn luyện H1 (seed 42)

**🎤 Nói:**
"Mạng có 4 đầu vào và **769 tham số**. Epoch tốt nhất là epoch 2, tức là mạng học xong rất nhanh. Điều này hợp lý vì chỉ có 16 tổ hợp đầu vào. ATE ra **+0.2453**, gần như trùng với B0."

**⚠️ Lưu ý:** nếu thầy hỏi "epoch 2 có phải là chưa học xong không?", trả lời: với $2^4 = 16$ cấu hình, mỗi cấu hình có hàng trăm mẫu, nên MLP gần như chỉ cần học lại tần suất của từng ô. Dừng sớm là đúng.

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

### Cell 49 · code · Huấn luyện H2 (seed 42)

**🎤 Nói:**
"Bảng thứ nhất là bước chọn $\lambda_1$ trên VALIDATION. $10^{-4}$ thắng (−2251 so với −2277), nên được đóng băng cho mọi seed.

Bảng thứ hai là bước chọn ngưỡng. $\tau = 0.1$ cho 8 cạnh với likelihood tốt nhất. Thầy để ý dòng $\tau = 0.8$: đồ thị chỉ còn 1 cạnh và phải *ép* thêm $T \to Y$. Ngược lại, ở ngưỡng được chọn, cạnh $T \to Y$ được **học ra thật**, không phải ép.

Ràng buộc chu trình $h(W) = 3.3 \times 10^{-7}$, tức là đã hội tụ. Trọng số lớn nhất ở các ô bị cấm đúng bằng 0. ATE là **+0.2428**."

### Cell 50 · markdown · 8.E Giao thức đánh giá 5 run

**🎤 Nói:**
"Đánh giá tách làm hai phần rõ ràng.

**Table A** là dự báo, *có dùng* $Y_{\text{test}}$: AUROC, Macro-F1, Brier. Ngưỡng F1 được chọn trên VAL.

**Table B** là can thiệp, *tuyệt đối không dùng* $Y_{\text{test}}$, chỉ chuẩn hóa trên $C_{\text{test}}$.

Khoảng tin cậy là percentile bootstrap 1000 lần trên các hồ sơ test, với mô hình được giữ cố định."

**📘 Từ khóa mới:**
* **AUROC:** xác suất để mô hình xếp một người tốt nghiệp *cao hơn* một người bỏ học. 0.5 là đoán bừa, 1.0 là hoàn hảo.
* **Macro-F1:** trung bình F1 của hai lớp. F1 là trung bình điều hòa của precision và recall.
* **Brier score:** $\frac1n\sum(p_i - y_i)^2$, sai số bình phương của xác suất dự đoán. Càng nhỏ càng tốt, và nó đo cả độ *hiệu chỉnh* (calibration).
* **$\Delta_{\text{Model}-B_0}$:** hiệu ghép cặp, tính trong từng seed rồi lấy trung bình.
* **Percentile bootstrap CI:** lấy mẫu có hoàn lại 545 hồ sơ, 1000 lần, tính ATE mỗi lần. Khoảng 2.5%–97.5% của 1000 giá trị đó là CI 95%. **Chỉ** phản ánh bất định do lấy mẫu quần thể tham chiếu, chưa gồm bất định của tham số hay của đồ thị.

### Cell 51 · code · Hàm đánh giá dùng chung

**🎤 Nói:**
"Cell này định nghĩa bốn hàm: chỉ số dự báo, CI bootstrap, gọi DoWhy để *identify trên chính đồ thị đã học*, và báo cáo overlap theo từng hồ sơ test. Nó cũng in phần cứng thực tế: CPU Intel 12 luồng, không có GPU."

### Cell 52 · code · Chạy 5 run ghép cặp

**🎤 Nói:**
"Đây là vòng lặp chính. Thầy nhìn dòng tóm tắt của từng seed: ATE của cả ba mô hình đều nằm trong khoảng **0.22 đến 0.25**. B0 luôn chọn K2, với 10–12 cạnh. H2 hội tụ ở cả 5 seed."

### Cell 53 · code · 8.E2 Quá trình huấn luyện, kiểm tra overfit, có / không early stopping

**🎤 Nói:**
"Trước khi xem kết quả, em muốn thầy thấy quá trình huấn luyện có ổn không. Ba tập được dùng tách bạch: TRAIN để khớp mô hình, VALIDATION để theo dõi sau mỗi epoch và chọn cấu hình, còn TEST chỉ dùng một lần ở cuối.

Ở Hình 8.2, panel (a) là loss của H1 qua từng epoch, nét liền là TRAIN, nét đứt là VAL. Loss VAL chạm đáy rất sớm, khoảng epoch 2 đến 4, nên early stopping dừng sau 17 đến 39 epoch và lấy lại trọng số tốt nhất, là chấm tròn. Panel (b) là H2: ràng buộc phi chu trình $h(W)$ giảm đều theo thang log và xuống dưới $10^{-6}$ sau 7–8 vòng. Panel (c) cho thấy sai số tái tạo tăng nhẹ, vì phạt chu trình càng chặt thì mô hình phải hy sinh một chút độ khớp để là một DAG hợp lệ.

Bảng 8.E2 trả lời câu hỏi có overfit không. AUROC của B0 và H1 trên TRAIN, VAL và TEST gần như bằng nhau, chênh TRAIN trừ TEST chỉ khoảng 0.002 và gần 0. H2 chênh lớn hơn một chút, khoảng 0.008, nhưng vẫn nhỏ hơn độ dao động giữa các seed. Vậy không có overfit đáng kể.

Cuối cùng, em huấn luyện H1 thêm một bản không có early stopping, chạy đủ 150 epoch, và lưu thành checkpoint riêng. Hình 8.2b cho thấy loss VAL không giảm thêm sau vài epoch đầu, trong khi TRAIN vẫn giảm chậm. Bảng 8.E2b xác nhận: bản không dừng sớm có loss VAL và loss TEST cao hơn, AUROC TEST thấp hơn một chút. Đó là dấu hiệu bắt đầu overfit, và early stopping đã chặn đúng chỗ."

**📘 Từ khóa mới:**
* **Overfit (quá khớp):** mô hình học thuộc cả nhiễu của TRAIN, nên điểm trên TRAIN tốt nhưng trên dữ liệu mới (VAL, TEST) kém đi. Dấu hiệu: khoảng cách giữa TRAIN và TEST lớn.
* **Loss curve (đường cong loss):** loss theo từng epoch. Nếu loss TRAIN tiếp tục giảm mà loss VAL đi ngang hoặc tăng, mô hình đang bắt đầu overfit.
* **Log-loss:** chính là BCE tính trên một tập dữ liệu. Nó phạt nặng những dự báo *tự tin mà sai*, nên nhạy với overfit hơn AUROC.
* **Accuracy:** tỷ lệ dự báo đúng khi lấy ngưỡng 0.5.

**⚠️ Lưu ý:**
* Nếu thầy hỏi *"sao loss TRAIN lại cao hơn VAL?"* — loss TRAIN được đo trong lúc huấn luyện, khi dropout 0.2 đang bật, nên bị đội lên. Loss VAL được đo ở chế độ đánh giá, không có dropout.
* Nếu thầy hỏi *"sao hội tụ sớm vậy, có underfit không?"* — cha của $Y$ là bốn biến nhị phân, đầu vào chỉ có $2^4 = 16$ tổ hợp, MLP chỉ cần học 16 xác suất. AUROC ngang B0 cho thấy nó học đủ, không thiếu.
* Bảng này **không** chọn hay chỉnh gì theo TEST. Nó chỉ đọc lại mô hình đã đóng băng. AUROC TEST trùng với Table A.
* Hai checkpoint: `h1_seed<s>_early_stop.pt` (bản dùng trong Tables A–C) và `h1_seed<s>_no_early_stop.pt` (bản chạy đủ 150 epoch, chỉ để so sánh).

### Cell 54 · markdown · 8.E2 Đọc Hình 8.2, 8.2b và Bảng 8.E2, 8.E2b

**🎤 Nói:**
"Em tóm lại: cả H1 lẫn H2 đều huấn luyện ổn định và hội tụ, không overfit đáng kể. Vì vậy, việc H1 và H2 không dự báo tốt hơn B0 *không phải* do huấn luyện hỏng. Giới hạn nằm ở dữ liệu: với ít biến cha rời rạc, bảng CPD của B0 đã biểu diễn được hết, mạng nơ-ron không còn gì để khai thác thêm. Early stopping không làm đổi kết luận nhân quả, ATE trung bình vẫn khoảng +0.24, nhưng giúp ước lượng ổn định hơn giữa các seed."

### Cell 55 · code · Xuất Table A, Table B

**🎤 Nói:**
"Cell này tổng hợp kết quả thành giá trị trung bình ± độ lệch chuẩn qua 5 run, rồi ghi ra CSV trong `artifacts/tables`."

### Cell 56 · markdown · 8.F Refuter và Table C

**🎤 Nói:**
"Table C có hai refuter bắt buộc là placebo và subset. Em nói rõ *đối tượng được kiểm tra*. Với B0 và H2, refuter của DoWhy chạy trên *đồ thị* của mô hình, với bộ ước lượng tuyến tính. Với H1, refuter tuyến tính không áp dụng được, nên em tự cài một kiểm định *huấn luyện lại*: hoán vị $T$ rồi huấn luyện lại MLP, và huấn luyện lại trên 80% TRAIN. Kiểm định này được ghi nhãn riêng trong bảng."

**📘 Từ khóa mới:**
* **Paired spread:** $\max(\text{ATE}) - \min(\text{ATE})$ qua 5 seed. Đo xem thuật toán dao động bao nhiêu khi chỉ đổi cách chia dữ liệu.

### Cell 57 · code · Chạy refuter, xuất Table C

**🎤 Nói:**
"Refuter chạy trên seed 42. Kết quả em đọc ở cell kế tiếp."

### Cell 58 · code · Hiển thị Tables A, B, C, H3, Multi-Treatment

**🎤 Nói:**
"Cell này in tất cả các bảng. Em tóm tắt bằng lời, chi tiết hơn ở mục 8.3:
* **Table A:** AUROC của ba mô hình đều khoảng 0.80.
* **Table B:** ATE khoảng 0.24 ở cả ba mô hình.
* **Table C:** placebo về khoảng 0, subset ổn định. Trên đồ thị B0, DoWhy chọn tập điều chỉnh tối thiểu là {học phí, tuổi}.
* **H3** là phụ lục, dùng đồ thị miền cố định.
* **Multi-Treatment** liệt kê ba can thiệp $T_1, T_2, T_3$ theo từng bộ ước lượng và từng seed."

**📘 Từ khóa mới:**
* **Tập điều chỉnh tối thiểu:** trên đồ thị B0 học được, chỉ cần điều chỉnh theo {học phí, tuổi} là đã chặn hết đường cửa sau, vì các biến nền khác không đi vào $T$ trực tiếp.

### Cell 59 · code · Bảng positivity $T_1 \times T_2$

**🎤 Nói:**
"Bảng này là lý do em không tin $T_2$. Trong 449 sinh viên không có học bổng và không đóng phí đúng hạn, chỉ **4.5%** tốt nghiệp. Ô *có học bổng nhưng không đóng phí đúng hạn* chỉ có **37 sinh viên**."

### Cell 60 · markdown · 8.2b Giới hạn của $T_2$, $T_3$

**🎤 Nói:**
"Con số 4.5% phù hợp với câu chuyện *bỏ học rồi mới ngừng đóng phí* hơn là *nợ phí nên bỏ học*. Vì vậy $T_2$ có dấu hiệu nhân quả ngược, và em không diễn giải nó là hiệu ứng của một chính sách xóa nợ.

$T_3$ thì thiếu positivity, vì chỉ có 37 sinh viên ở ô quan trọng. Hơn nữa, ATE của $T_3$ xấp xỉ bằng tổng của $T_1$ và $T_2$, nên không có bằng chứng về hiệu ứng *cộng hưởng*."

### Cell 61 · code · Hình 8.1 + bảng tóm tắt đồ thị 5 seed

**🎤 Nói:**
"Bảng tóm tắt cho thấy B0 rất ổn định: K2 ở cả 5 seed, 10–12 cạnh, và cha của $Y$ luôn là {T, Debtor, học phí, tuổi}. H2 thưa hơn, 7–9 cạnh, $h(W)$ trong khoảng $10^{-7}$, không có seed nào phải ép cạnh $T \to Y$. Ở 4 trên 5 seed, H2 thêm điểm vào làm cha của $Y$.

Hình 8.1 lấy seed 42 làm ví dụ. Hai ô trên là mặt nạ $M$ dùng chung và ma trận trọng số $|W|$ của H2; ô có khung đỏ là cạnh được giữ vì vượt ngưỡng $\tau$. Hai ô dưới vẽ đồ thị của B0 và H2 theo cùng bố cục: các biến nền xếp thành cột bên trái, học bổng ở hàng dưới, tốt nghiệp bên phải. Nhờ vậy thầy thấy ngay khác biệt: B0 học thêm một chuỗi cạnh giữa các biến nền với nhau, còn H2 gần như chỉ giữ các cạnh đi vào $T$ và $Y$."

**⚠️ Lưu ý:** cột biến nền trong hai đồ thị dưới được xếp theo thứ tự Z2, Z3, Z5, Z4, Z1, chứ không theo Z1 → Z5, để các cạnh giữa các biến nền mà B0 học được nối hai nút kề nhau và không đè lên nhau. Nếu thầy hỏi, chỉ cần nói: "thứ tự xếp chỉ để hình dễ đọc, không mang ý nghĩa thời gian".

### Cell 62 · code · Tính $q_m(t, C_i)$ minh bạch

**🎤 Nói:**
"Cell này giúp thầy *thấy tận mắt* công thức. Em nạp lại B0 và checkpoint của H1, rồi in $q(0)$ và $q(1)$ cho vài hồ sơ test.

Ví dụ dòng đầu: sinh viên không nợ, không đóng phí đúng hạn, trên 20 tuổi. Nếu không có học bổng thì xác suất tốt nghiệp chỉ khoảng 1%. Nếu có học bổng thì lên khoảng 27%.

Lấy trung bình trên 545 hồ sơ thì ra risk 0.540 và 0.786, ATE +0.2467, khớp đúng với cell 45. Và em nhấn mạnh: **cột $Y_{\text{test}}$ không hề được dùng.**"

### Cell 63 · markdown · 8.2c Đọc Hình 8.1

**🎤 Nói:**
"Ba ý chính.

Một, **trọng số cạnh không phải là hiệu ứng nhân quả**. $|W|$ chỉ là đại lượng nội bộ của quá trình học. Hiệu ứng được tính lại bằng truy vấn can thiệp.

Hai, chiều cạnh *giữa các biến nền* thay đổi giữa các seed. Điều này không ảnh hưởng tới $do(T)$, vì mọi $Z$ đều có trước $T$. Nhưng nó cho thấy dữ liệu không xác định được chiều giữa các biến cùng tầng.

Ba, vì H2 thêm điểm vào (3 trạng thái) vào tập cha của $Y$, khoảng 1–2% hồ sơ test thiếu overlap. Ở những hồ sơ đó, ước lượng dựa vào làm mịn BDeu."

### Cell 64 · code · Hình 8.3 — forest plot ATE

**🎤 Nói:**
"Hình này đặt ATE của từng mô hình ở từng seed, kèm CI, cạnh đường chênh lệch thô 34.54%. Mọi điểm đều tụ quanh 0.24 và đều nằm xa bên trái đường chênh lệch thô."

**📘 Từ khóa mới:**
* **Forest plot:** mỗi dòng là một ước lượng, gồm một chấm và một thanh CI. Dùng để so sánh nhanh nhiều ước lượng.

### Cell 65 · markdown · 8.3 Phân tích kết quả

**🎤 Nói:**
"Đây là kết luận của phần chính.

**Table A:** AUROC của B0, H1, H2 lần lượt là 0.7996, 0.8012 và 0.8034. Hiệu ghép cặp của H1 so với B0 là +0.0016 ± 0.0025. Hiệu này *nhỏ hơn* độ dao động giữa các run, nên **không có bằng chứng** H1 hay H2 dự báo tốt hơn. Đây là một null finding, và em báo cáo trung thực như vậy.

Macro-F1 của B0 và H1 trùng nhau vì chỉ có 16 cấu hình đầu vào. Ở ngưỡng đã chọn, hai mô hình gán nhãn giống hệt nhau.

**Table B:** ATE là +0.2418, +0.2374 và +0.2358. Kết luận can thiệp **không đổi** giữa ba mô hình. So với chênh lệch thô 34.54%, khoảng 10.4 điểm là thiên lệch do nhiễu quan sát được.

**Table C:** mọi refuter đều đạt. H2 đắt nhất, khoảng 14–20 giây mỗi run tùy máy và lần chạy, so với chưa tới nửa giây của B0.

**Multi-treatment:** $T_1$ cho khoảng 21.6–23.1% qua ba bộ ước lượng. $T_2$ và $T_3$ ra số lớn nhưng không được diễn giải là hiệu ứng chính sách."

---

## MỤC 9 — KẾT CỤC ĐA CHIỀU, TRUNG GIAN, CATE, ĐỘ NHẠY

### Cell 66 · markdown · Mở đầu mục 9

**🎤 Nói:**
"Mục 9 trả lời RQ4 về độ nhạy, và RQ5 về cơ chế cùng nhóm hưởng lợi. Em lưu ý trước: phần này dùng hồi quy có điều chỉnh trên *toàn bộ* quần thể, với 7 biến nền, khác với 5 biến rời rạc của Tables A–C. Vì vậy em coi đây là phân tích *bổ sung*."

### Cell 67 · markdown · Chú thích mã mục 9.A–9.G

**🎤 Nói:** *(lướt qua)* "Các cell tiếp theo sinh hai bảng: Alternative Outcomes và Heterogeneous CATE."

### Cell 68 · markdown · 9.A Hai quần thể

**🎤 Nói:**
"Em dùng hai quần thể. Quần thể 3,630 sinh viên đã có kết cục, dùng cho điểm số, tín chỉ, trung gian và CATE. Quần thể đầy đủ 4,424 sinh viên, *giữ lại* nhóm Enrolled, dùng cho tỷ lệ giữ chân."

### Cell 69 · code · Nạp hai quần thể

**🎤 Nói:**
"Phân bố gốc: 2,209 tốt nghiệp, 1,421 bỏ học, 794 đang học. Hai quần thể có đúng kích thước 3,630 và 4,424."

### Cell 70 · markdown · 9.B Kết cục điểm HK2

**🎤 Nói:**
"Kết cục thứ nhất là điểm trung bình học kỳ 2, trên thang 20. Em hồi quy điểm theo $T$ và $\mathbf{Z}$. Hệ số của $T$ chính là ATE."

**📘 Từ khóa mới:**
* **Định lý Frisch–Waugh–Lovell:** hệ số của $T$ trong hồi quy bội bằng hệ số khi hồi quy "phần dư của $Y$" theo "phần dư của $T$", sau khi đã loại ảnh hưởng của $\mathbf{Z}$. Nói gọn: hệ số đó là tác động của $T$ *khi giữ $\mathbf{Z}$ cố định*.

### Cell 71 · code · Ước lượng điểm HK2

**🎤 Nói:**
"Học bổng làm điểm HK2 tăng **+1.48 điểm** trên thang 20, CI từ 1.09 đến 1.86. Chênh lệch thô là 2.63 điểm, nên sau điều chỉnh hiệu ứng giảm gần một nửa. Đây lại là dấu vết của nhiễu."

### Cell 72 · markdown · 9.C Tỷ lệ đỗ tín chỉ

**🎤 Nói:**
"Kết cục thứ hai là tỷ lệ tín chỉ đỗ trên tín chỉ đăng ký ở HK2, nằm trong khoảng 0 đến 1. Chỉ số này đo xem sinh viên có theo kịp tiến độ hay không."

### Cell 73 · code · Ước lượng tỷ lệ đỗ tín chỉ

**🎤 Nói:**
"Tỷ lệ đỗ tín chỉ tăng **+14.5 điểm phần trăm**, CI từ 11.8 đến 17.2. Tỷ lệ sinh viên đỗ hết mọi môn là 33.9% ở nhóm không học bổng và 58.2% ở nhóm có học bổng."

**⚠️ Lưu ý:** điểm HK2 và tín chỉ HK2 là kết cục *sau* $T$. Ở đây chúng được dùng làm *kết cục*, không phải biến điều chỉnh, nên hợp lệ.

### Cell 74 · markdown · 9.D Giữ chân toàn trường (AIPW)

**🎤 Nói:**
"Đây là câu trả lời cho câu hỏi *bỏ Enrolled có làm sai lệch kết quả không*. Em giữ cả 4,424 sinh viên và định nghĩa *giữ chân* là tốt nghiệp hoặc đang học, so với bỏ học. Ước lượng bằng AIPW, một bộ ước lượng bền vững kép."

**📘 Từ khóa mới:**
* **AIPW — Augmented Inverse Probability Weighting, bền vững kép (doubly robust):**
  $$\widehat{\text{ATE}}_{DR} = \frac1N\sum_i\Big[\hat\mu_1(Z_i) + \tfrac{T_i(Y_i - \hat\mu_1(Z_i))}{\hat e(Z_i)}\Big] - \Big[\hat\mu_0(Z_i) + \tfrac{(1-T_i)(Y_i - \hat\mu_0(Z_i))}{1-\hat e(Z_i)}\Big]$$
  Công thức kết hợp mô hình kết cục $\hat\mu_t$ với propensity $\hat e$. Ước lượng vẫn nhất quán nếu **một trong hai** mô hình đúng, đó là ý nghĩa của "bền vững kép".

### Cell 75 · code · Ước lượng giữ chân

**🎤 Nói:**
"Tỷ lệ giữ chân là 64.3% nếu không cấp và 81.2% nếu cấp. ATE là **+16.87%**, CI bootstrap từ 13.8 đến 19.7. Hiệu ứng vẫn dương rõ khi đưa nhóm Enrolled vào. Như vậy việc loại Enrolled ở phân tích chính không tạo ra kết luận."

### Cell 76 · markdown · 9.E Trung gian

**🎤 Nói:**
"Học bổng tác động qua kênh nào? Em tách tổng hiệu ứng thành hai phần. Phần *trực tiếp* có thể là do bớt áp lực tài chính, không phải đi làm thêm. Phần *gián tiếp* đi qua việc đỗ được nhiều tín chỉ ở HK1, tức là có đà học tập sớm."

**📘 Từ khóa mới:**
* **TE = NDE + NIE:**
  * **TE (Total Effect)**, tổng hiệu ứng.
  * **NDE (Natural Direct Effect)**, hiệu ứng trực tiếp tự nhiên: đổi $T$ nhưng *giữ* $M$ ở mức nó sẽ có khi $T=0$.
  * **NIE (Natural Indirect Effect)**, hiệu ứng gián tiếp: giữ $T$, chỉ đổi $M$ từ mức của $T=0$ sang mức của $T=1$.
* **Academic momentum:** đà học tập. Qua được kỳ đầu thì dễ đi tiếp.

### Cell 77 · code · Ước lượng NDE và NIE

**🎤 Nói:**
"Học bổng làm tăng 0.73 tín chỉ đỗ ở HK1. Tổng hiệu ứng là +21.61%, trong đó trực tiếp **+16.96%** (78.5%) và gián tiếp qua HK1 **+4.65%** (21.5%). Như vậy khoảng một phần năm tác dụng của học bổng đi qua kết quả học kỳ đầu."

**⚠️ Lưu ý:** cách tính là tích hệ số của hai hồi quy tuyến tính. Cách này cần thêm một giả định: *không có nhiễu giữa M và Y*. Giả định này mạnh hơn giả định của ATE, nên nói rõ khi bị hỏi.

### Cell 78 · markdown · 9.F CATE

**🎤 Nói:**
"Nhóm nào hưởng lợi nhiều nhất? Em chia theo ba đặc trưng định trước: tuổi, nợ học phí, và nơi ở. Trong từng nhóm, em khớp riêng một mô hình G-computation logistic. Em nói rõ đây *không* phải Causal Forest, và chưa có CI theo nhóm."

**📘 Từ khóa mới:**
* **CATE — Conditional ATE:**
  $$\text{CATE}(x) = \mathbb{E}[Y \mid do(T=1), X=x] - \mathbb{E}[Y \mid do(T=0), X=x]$$
  Đây là ATE nhưng tính trong nhóm $X=x$.
* **G-computation:** khớp mô hình $\hat{\mathbb{E}}[Y \mid T, \mathbf{Z}]$, dự đoán cho mọi người với $T=1$ rồi với $T=0$, lấy trung bình và trừ. Đây là cách làm *tương tự* $q_m$ ở mục 8, nhưng dùng hồi quy logistic thay cho mạng Bayes.

### Cell 79 · code · Bảng CATE

**🎤 Nói:**
"Nhóm trên 20 tuổi có CATE **+28.0%**, so với +16.1% ở nhóm trẻ, chênh nhau khoảng 12 điểm. Nhóm ở cùng gia đình có +25.9%, nhóm thuê trọ có +17.6%.

Với nhóm nợ học phí, hiệu ứng tuyệt đối +20.3% **không lớn hơn** nhóm không nợ (+21.4%). Chỉ có tỷ số rủi ro là lớn hơn, 2.03 lần, vì mức nền của nhóm này rất thấp, khoảng 19.7%."

**📘 Từ khóa mới:**
* **Relative Risk (tỷ số rủi ro):** $\text{Risk}(do(1))/\text{Risk}(do(0))$. Con số 2.03x nghĩa là *xác suất tốt nghiệp gấp 2.03 lần*. Nó **không** có nghĩa là hưởng lợi gấp 2 lần. Tỷ số lớn khi mức nền thấp.

### Cell 80 · markdown · 9.G Tổng hợp

**🎤 Nói:** "Bảng tiếp theo gom bốn kết cục mở rộng vào một chỗ và lưu ra CSV."

### Cell 81 · code · Bảng tổng hợp kết cục mở rộng

**🎤 Nói:**
"Bốn chiều kết cục đều cùng hướng dương: điểm HK2 +1.48, tín chỉ +14.5%, giữ chân +16.9%, và trung gian 21.5%. Kết luận *học bổng có lợi* không phụ thuộc vào việc chọn kết cục nào."

### Cell 82 · markdown · 9.H Độ nhạy Cinelli–Hazlett

**🎤 Nói:**
"Đây là phần trả lời cho giả định *không có nhiễu ẩn*, thứ mà refuter không kiểm tra được. Câu hỏi là: *một biến nhiễu ẩn phải mạnh cỡ nào thì mới xóa sạch được hiệu ứng?*"

**📘 Từ khóa mới:**
* **Partial $R^2$:** phần phương sai *còn lại* (sau khi đã tính các biến khác) mà một biến giải thích được.
* **Robustness Value (Cinelli & Hazlett, 2020):**
  $$f_q = \frac{q\,|t_T|}{\sqrt{df}}, \qquad RV_q = \tfrac12\Big(\sqrt{f_q^4 + 4f_q^2} - f_q^2\Big)$$
  $t_T$ là thống kê t của hệ số học bổng, và $df$ là bậc tự do. $RV_{q=1}$ là mức partial $R^2$ tối thiểu mà một nhiễu ẩn phải có với **cả** $T$ **và** $Y$ để kéo ước lượng về 0. Với $RV_{q=1,\alpha=0.05}$, mục tiêu là làm mất ý nghĩa thống kê ở mức 5%.
* **Benchmark:** so sánh RV với partial $R^2$ của các biến *quan sát được*, để có cảm giác "nhiễu ẩn phải mạnh hơn biến X bao nhiêu lần".

### Cell 83 · code · Tính RV

**🎤 Nói:**
"Hệ số học bổng là +0.227, với t = 14.2. Từ đó **RV bằng 20.97%**. Nghĩa là một nhiễu ẩn phải giải thích ít nhất khoảng 21% phương sai dư của *cả* học bổng *lẫn* tốt nghiệp thì mới xóa được hiệu ứng.

Để so sánh, biến quan sát mạnh nhất là học phí đúng hạn, chỉ có 10.4%. Tuổi có 4.2%. Các biến còn lại đều dưới 1%."

### Cell 84 · markdown · 9.4 Đọc kết quả

**🎤 Nói:**
"Kết luận về độ nhạy: hiệu ứng vững trước những nhiễu ẩn **yếu hơn** biến học phí. Nhưng nó **không miễn nhiễm** với một nhiễu ẩn mạnh gấp đôi biến học phí. Đó là cách em diễn đạt trung thực.

Em cũng ghi lại một lỗi đã sửa. Bản trước dùng công thức xấp xỉ sai, ra 23.40%. Notebook hiện dùng đúng công thức Cinelli–Hazlett, ra 20.97%."

---

## MỤC 10 — PHÂN BỔ CHÍNH SÁCH VÀ CHUYỂN DỊCH NGOẠI MIỀN

### Cell 85 · markdown · 10.1–10.2 Thiết kế

**🎤 Nói:**
"Câu hỏi RQ6: giữ nguyên ngân sách 969 suất, nhưng cấp theo *ai hưởng lợi nhiều nhất*, tức là theo CATE cá nhân, thay vì theo cách hiện tại, thì có thêm bao nhiêu người tốt nghiệp?

Phần 10.2 là phụ lục: thử chuyển mô hình sang Kaggle S4E6."

**📘 Từ khóa mới:**
* **Policy learning (học chính sách):** chọn quy tắc phân bổ can thiệp sao cho tối đa hóa kết cục chung, trong giới hạn ngân sách.
* **OOD — Out-of-distribution / chuyển dịch ngoại miền:** áp mô hình lên một quần thể khác với quần thể đã học.
* **Kaggle S4E6:** bộ dữ liệu **tổng hợp** (synthetic) sinh từ UCI 697, có 76,518 dòng. Nó không phải một trường thật khác.

### Cell 86 · code · Bảng 10.1 và 10.2

**🎤 Nói:**
"Nếu xếp hạng theo CATE, tỷ lệ tốt nghiệp *dự kiến* tăng từ 60.85% lên 65.37%, tức khoảng **+164 sinh viên** trên toàn bộ 3,630 hồ sơ. Để so sánh, ưu tiên theo nợ phí chỉ thêm 49, còn phân bổ ngẫu nhiên thêm 37.

Bảng 10.2: mô hình học trên UCI cho ATE gần như giống nhau khi áp lên UCI (0.215) và lên Kaggle (0.214). Còn mô hình học trên Kaggle cho ATE khoảng 0.30–0.32."

### Cell 87 · markdown · 10.3 Đọc kết quả

**🎤 Nói:**
"Em nêu thẳng ba giới hạn của con số +164.

Một, đây là ước lượng **in-sample**: cùng một mô hình vừa dùng để *chọn* người vừa dùng để *chấm điểm* chính sách, nên con số có xu hướng lạc quan.

Hai, danh sách xếp hạng dùng biến học phí, chính là biến đáng ngờ ở mục 8.2b.

Ba, kết quả dựa trên giả định không có nhiễu ẩn.

Vì vậy em chỉ khuyến nghị **thí điểm có kiểm soát**, không khuyến nghị áp dụng ngay.

Về Kaggle: kết quả *trái chiều*, nên em không khẳng định được tính bất biến cấu trúc. Hơn nữa Kaggle là dữ liệu tổng hợp, nên đây không phải kiểm định trên một quần thể sinh viên thật khác."

**📘 Từ khóa mới:**
* **In-sample và cross-fitting:** cross-fitting là chia dữ liệu thành nhiều phần, dùng phần này để chọn và phần kia để chấm, nhằm tránh lạc quan quá mức. Đồ án chưa làm cross-fitting, và em ghi rõ là hạn chế.
* **Off-policy evaluation:** đánh giá một chính sách *chưa từng được áp dụng* bằng dữ liệu sinh ra từ chính sách cũ, ví dụ bằng IPW hoặc DR.

---

## MỤC 11 — TỔNG KẾT

### Cell 88 · markdown · 11.1 Trả lời RQ + 11.2 Khuyến nghị

**🎤 Nói:**
"Em quay lại bảng câu hỏi ở đầu bài.

* **RQ0:** dưới các giả định đã nêu, học bổng tăng xác suất tốt nghiệp **khoảng 24 điểm phần trăm**, ổn định qua cả ba mô hình và cả 5 seed.
* **RQ1:** đúng là chênh lệch thô bị thổi phồng. Khoảng 10 điểm, gần một phần ba, là do nhiễu.
* **RQ2 và RQ3:** phần học sâu **không** cải thiện dự báo một cách có ý nghĩa. H2 học được đồ thị hợp lệ và tự tìm ra cạnh $T \to Y$. Kết luận can thiệp không đổi. Em coi đây là một kết quả có giá trị: mô hình truyền thống B0 đã đủ tốt cho bài toán này.
* **RQ4:** kết luận khá vững. Refuter đạt, và RV bằng 21%, gấp đôi biến quan sát mạnh nhất. Nhưng nó không miễn nhiễm với nhiễu ẩn.
* **RQ5:** khoảng 21% tác động đi qua kết quả HK1. Nhóm lớn tuổi hưởng lợi nhiều hơn.
* **RQ6:** ủng hộ có điều kiện, cần thí điểm.

Hạn chế lớn nhất em tự nhận là biến học phí đúng hạn. Biến này có thể là hậu quả của việc bỏ học.

Bốn khuyến nghị: thí điểm phân bổ theo CATE, hỗ trợ học thuật sớm trong HK1 vì khoảng 21.5% tác động đi qua kỳ này, dùng refuter làm quy chuẩn kiểm định chính sách, và dùng mạng Bayes B0 làm lõi cho hệ thống cảnh báo sớm.

Em xin cảm ơn thầy và sẵn sàng nhận câu hỏi."

**⚠️ Lưu ý:** khuyến nghị 4 đã được viết lại cho khớp với null finding ở RQ2: B0 dự báo tương đương bản nơ-ron mà minh bạch hơn, nên B0 làm lõi cho hệ thống cảnh báo sớm. Nếu thầy hỏi "vậy làm H1/H2 để làm gì?", trả lời: để *kiểm chứng* rằng kết luận không phụ thuộc cách tham số hóa hay cách học đồ thị. Kết quả âm tính ấy chính là bằng chứng độ vững.

---

## PHỤ LỤC — CHỈ MỤC TỪ KHÓA (tìm nhanh nơi giải thích)

| Từ khóa | Giải thích ở cell | Từ khóa | Giải thích ở cell |
| :--- | :---: | :--- | :---: |
| PGM, B0/H1/H2, Tables A–C, CATE (1 dòng), trung gian (1 dòng) | 0 | Graph surgery | 31 |
| Tầng thời gian, mặt nạ $M$, NOTEARS, BDeu/ESS, $C_{\text{test}}$, $Y_{\text{test}}$ | 1 | Truncated factorization | 32 |
| Actionable, $do()$, confounder, mediator, CBN, HillClimb, BIC/BDeu/K2, TRAIN/VAL/TEST | 2 | Identifiability | 33 |
| **ATE**, null finding | 3 | DoWhy, LR, propensity score, refuter | 34 |
| Split | 5 | 3 loại refuter | 35 |
| Positivity / overlap | 7 | MLP, checkpoint, multi-treatment, nhân quả ngược, T-learner | 37 |
| Estimand, nhiễu ẩn, consistency | 8 | Stratified split, seed | 40 |
| Biến hậu can thiệp | 11 | Paired run | 43 |
| Rời rạc hóa | 12 | Bắt buộc $T \to Y$, log-likelihood VAL, K2, **$q_m(t, C_i)$**, risk | 44 |
| Mutual Information | 13 | BCE, early stopping, BatchNorm/Dropout | 46 |
| $P(Y \mid T)$ vs $P(Y \mid do(T))$, confounding bias, collider | 16 | $h(W)$, Augmented Lagrangian, $\lambda_1$, $\tau$ | 48 |
| Naive gap | 17 | AUROC, Macro-F1, Brier, $\Delta$, bootstrap CI | 50 |
| DAG, PC/FCI, kiểm định độc lập ($\chi^2$, Fisher-Z), v-structure, Meek, CPDAG, PAG, GES, cực trị địa phương, NP-hard, $h(W)$, lệch phương sai | 20 | Paired spread | 56 |
| `ExpertKnowledge` | 21 | Tập điều chỉnh tối thiểu | 58 |
| CPD, $\text{Pa}(X)$, phân rã, MLE, EM, Neural CPD | 24 | Forest plot | 64 |
| Backdoor path, d-separation, **công thức điều chỉnh cửa sau** | 27 | FWL | 70 |
| Domain DAG | 29 | AIPW, doubly robust | 74 |
| TE / NDE / NIE | 76 | CATE (đầy đủ), G-computation | 78 |
| Relative Risk | 79 | Partial $R^2$, **RV**, benchmark | 82 |
| Policy learning, OOD, Kaggle S4E6 | 85 | In-sample, cross-fitting, off-policy | 87 |
| Overfit, loss curve, log-loss, accuracy | 53 | | |
