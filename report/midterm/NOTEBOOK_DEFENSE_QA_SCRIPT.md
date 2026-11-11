# KỊCH BẢN ĐỐI ĐÁP PHẢN BIỆN TRỰC TIẾP TRÊN JUPYTER NOTEBOOK (MIDTERM DEFENSE)
> **Tài liệu phục vụ:** Buổi bảo vệ Giữa kỳ trực tuyến môn *Mô hình đồ thị có xác suất (IT701100)*  
> **Giảng viên hướng dẫn & chấm thi:** TS. Nguyễn Chí Thiện  
> **Hình thức trình bày:** Trình diễn và đối soát trực tiếp trên file [`00_MASTER_Causal_PGM_Educational_Intervention.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/00_MASTER_Causal_PGM_Educational_Intervention.ipynb)  
> **Quy chuẩn đối soát:** Slide 39–44, 51, 53–75, 76–77, 79, 80 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).  
> **Lời trình bày từng cell (kèm giải thích từ khóa ở lần xuất hiện đầu):** [`KICH_BAN_TRINH_BAY_NOTEBOOK.md`](KICH_BAN_TRINH_BAY_NOTEBOOK.md). File này chỉ chứa các vòng **chất vấn**.

---

## MỤC LỤC KỊCH BẢN CÁC VÒNG CHẤT VẤN (TỪ CỐT LÕI ĐẾN MỞ RỘNG CHUYÊN SÂU)

- [VÒNG 0: MỞ ĐẦU — GIẢI TRÌNH VỀ VIỆC SỬ DỤNG AI SINH MÃ NGUỒN](#vòng-0-mở-đầu--giải-trình-về-việc-sử-dụng-ai-sinh-mã-nguồn)
- [VÒNG 1: MỤC 1 — KIỂM TOÁN DỮ LIỆU & CHỐNG RÒ RỈ BIẾN THỜI GIAN](#vòng-1-mục-1--kiểm-toán-dữ-liệu--chống-rò-rỉ-biến-thời-gian)
- [VÒNG 1b: MỤC 1.4–1.5 — VÌ SAO LOẠI 794 SINH VIÊN `ENROLLED` KHỎI Y](#vòng-1b-mục-1415--vì-sao-loại-794-sinh-viên-enrolled-khỏi-y)
- [VÒNG 1c: MỤC 1.6 — HÌNH 1.2 (MUTUAL INFORMATION): VÌ SAO XANH DÙNG, ĐỎ, XÁM KHÔNG DÙNG](#vòng-1c-mục-16--hình-12-mutual-information-vì-sao-xanh-dùng-đỏ-xám-không-dùng)
- [VÒNG 2: MỤC 2.3 — NGHỊCH LÝ QUAN SÁT THÔ P(Y|T) VS CAN THIỆP DO(T)](#vòng-2-mục-23--nghịch-lý-quan-sát-thô-pyt-vs-can-thiệp-dot)
- [VÒNG 2b: MỤC 2 — HÌNH 2.1: Ô ĐỎ "T=1, CHƯA ĐÓNG HỌC PHÍ" CHỈ CÓ 37 SINH VIÊN](#vòng-2b-mục-2--hình-21-ô-đỏ-t1-chưa-đóng-học-phí-chỉ-có-37-sinh-viên)
- [VÒNG 3: B0, H1, H2 — HỌC CẤU TRÚC, MẶT NẠ M VÀ ĐỊNH DANH](#vòng-3-b0-h1-h2--học-cấu-trúc-mặt-nạ-m-và-định-danh)
- [VÒNG 4: MỤC 4 — BAYESIAN SMOOTHING (BDEU) VS MAXIMUM LIKELIHOOD (MLE)](#vòng-4-mục-4--bayesian-smoothing-bdeu-vs-maximum-likelihood-mle)
- [VÒNG 5: MỤC 7 — KHUNG LÀM VIỆC DOWHY 4 BƯỚC & 2 BÀI TEST BÁC BỎ](#vòng-5-mục-7--khung-làm-việc-dowhy-4-bước--2-bài-test-bác-bỏ)
- [VÒNG 6: BẢNG A, B, C — SO SÁNH B0 / H1 / H2 VÀ QUY TẮC TÍNH ATE](#vòng-6-bảng-a-b-c--so-sánh-b0--h1--h2-và-quy-tắc-tính-ate)
- [VÒNG 6b: MỤC 8.E2 — QUÁ TRÌNH HUẤN LUYỆN, EPOCH, OVERFIT VÀ EARLY STOPPING](#vòng-6b-mục-8e2--quá-trình-huấn-luyện-epoch-overfit-và-early-stopping)
- [VÒNG 7: MỤC 8.2 & BẢNG MULTI-TREATMENT — MỞ RỘNG CAN THIỆP (T1, T2, T3) VÀ CHẨN ĐOÁN H2](#vòng-7-mục-82--bảng-multi-treatment--mở-rộng-can-thiệp-t1-t2-t3-và-chẩn-đoán-h2)
- [VÒNG 8: MỤC 9–10 — CATE, PHÂN BỔ HỌC BỔNG (+164 SV) & CHUYỂN DỊCH SANG KAGGLE](#vòng-8-mục-910--cate-phân-bổ-học-bổng-164-sv--chuyển-dịch-sang-kaggle)

---

## VÒNG 0: MỞ ĐẦU — GIẢI TRÌNH VỀ VIỆC SỬ DỤNG AI SINH MÃ NGUỒN

### 🎯 Tình huống
Thầy Thiện mở notebook lên, thấy cấu trúc code sạch đẹp, chú thích tiếng Việt chuyên nghiệp, sử dụng nhiều hàm toán học chuẩn chỉ và hỏi thẳng:
> **Thầy Thiện:**  
> *"Code này thầy thấy định dạng rất trau chuốt, hàm số và cấu trúc chuẩn mực. Có phải toàn bộ code này là do AI sinh ra không? Nhóm em tự viết được bao nhiêu %, và làm sao thầy biết các em thực sự hiểu mã nguồn này mà không phải là 'vibe coding'?"*

### 💡 Ý đồ của Thầy
Thầy kiểm tra tính trung thực học thuật (Academic Integrity theo Slide 49, 78) và muốn biết sinh viên có làm chủ kiến trúc hệ thống hay chỉ phó mặc cho AI.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, nhóm em xin phép minh bạch 100%: Nhóm có chủ động sử dụng AI (như Claude Code / Gemini) đóng vai trò là **trợ lý lập trình (AI Pair Programmer)** để tăng tốc độ gõ cú pháp và chuẩn hóa docstring tiếng Việt theo tinh thần khuyến khích của môn học.*  
> 
> *Tuy nhiên, AI tuyệt đối không thể tự thiết kế bài toán này nếu không có sự chỉ đạo của nhóm. Toàn bộ các quyết định cốt lõi gồm:*  
> 1. *Xây dựng đồ thị nhân quả phân cấp thời gian với 5 biến nền tảng $\mathbf{Z}$ và biến trung gian $M_1$ (mục 6).*  
> 2. *Thiết lập ma trận mặt nạ $M$ chặn các cạnh ngược thời gian trong NOTEARS ($W_{\text{eff}} = M \odot W$) và ràng buộc cấm cạnh (mục 3 và 8.D).*  
> 3. *Lựa chọn phân phối tiên nghiệm BDeu prior thay vì MLE để xử lý 96 ô điều kiện thưa: $ESS=5$ cho các mô hình đối chuẩn $B_0/H_2$, mục 4.3 minh họa thêm $ESS=10$.*  
> 4. *Quy tắc chuẩn hóa ATE trên tập tham chiếu kiểm thử $C_{\text{test}}$ độc lập ($n=545$) mà không đụng vào nhãn $Y_{\text{test}}$ (mục 8.E).*  
> 
> *Mọi dòng code do AI đề xuất đều phải vượt qua bộ 4 bài kiểm tra ràng buộc logic (mục 5). Nhóm em nắm tường tận từng biến số, tensor và sẵn sàng giải thích chi tiết bất kỳ dòng lệnh nào trên notebook này ạ!"*

### ⚠️ Lỗi chết người cần tránh
- **Tuyệt đối không chối:** "Dạ không, nhóm em tự gõ tay 100% không dùng AI" (Thầy nhìn phong cách code sẽ biết ngay và đánh trượt vì thiếu trung thực).
- **Tuyệt đối không thừa nhận thụ động:** "Dạ AI viết hết, em chỉ bấm Run thôi ạ".

---

## VÒNG 1: MỤC 1 — KIỂM TOÁN DỮ LIỆU & CHỐNG RÒ RỈ BIẾN THỜI GIAN

### 🎯 Vị trí trên Notebook
Mục 1.2–1.6: nạp và kiểm toán dữ liệu, từ điển biến, luồng loại trừ, kiểm toán bổ sung.

### ❓ Thầy Thiện chất vấn
> *"Thầy nhìn vào mục 1, dữ liệu UCI ID 697 có 4,424 dòng và 35 thuộc tính:*  
> 1. *Tại sao các em lại viết lệnh `df = df[df['Target'] != 'Enrolled']` vứt bỏ đi 794 sinh viên? Việc này có gây thiên lệch chọn mẫu (selection bias) không?*  
> 2. *Trong 35 cột của UCI, có sẵn cột điểm số và tín chỉ học kỳ 1 và học kỳ 2 (`Curricular units 1st sem`, `Curricular units 2nd sem`). Tương quan của chúng với việc tốt nghiệp cực kỳ mạnh. Tại sao các em lại bỏ hết ra ngoài, chỉ giữ lại 5 biến nền tảng $Z_1..Z_5$?"*

### 💡 Ý đồ của Thầy
Kiểm tra Tiêu chí 1 & 4 Rubric (Slide 54 & 55): Khả năng phân định biến can thiệp, biến kết quả và kiểm soát rò rỉ dữ liệu (No Data Leakage).

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, nhóm xin giải trình 2 quyết định thiết kế này:*  
> 
> *1. **Về việc lọc 794 sinh viên `Enrolled`:** Bài toán chính sách của trường đại học là đánh giá rủi ro bỏ học để can thiệp kịp thời. Do đó, nhãn kết quả $Y$ phải được định nghĩa nhị phân dứt khoát: **Tốt nghiệp (`Graduate` = 1) đối lập với Bỏ học (`Dropout` = 0)**. Nhóm `Enrolled` là những sinh viên vẫn đang theo học tại thời điểm khảo sát, kết cục của họ chưa diễn ra (right-censored). Nếu gán họ vào 0 hoặc 1 đều làm sai lệch bản chất biến mục tiêu. Quy trình lọc này tuân thủ đúng lưu đồ hợp lệ (Eligibility Flow) tại Slide 55, đưa cỡ mẫu chuẩn về 3,630 sinh viên.*  
> 
> *2. **Về việc loại bỏ điểm HK1/HK2 khỏi tập hiệp biến nền $Z$:** Đây là ranh giới sống còn giữa **Machine Learning tương quan** và **Suy diễn nhân quả**:*  
> - *Học bổng ($T$) được xét và trao tại **thời điểm nhập học**.*  
> - *Điểm số HK1/HK2 diễn ra **sau khi** sinh viên đã nhận học bổng. Do đó, điểm HK1/HK2 đóng vai trò là **biến trung gian (Mediator)** trên con đường dẫn truyền nhân quả $T \to \text{Điểm HK} \to Y$.*  
> - *Nếu ta đưa điểm HK1/HK2 vào tập điều chỉnh $Z$, ta sẽ phạm lỗi **kiểm soát biến trung gian (Post-treatment conditioning)**, làm triệt tiêu mất hiệu ứng thực sự của học bổng lên tỷ lệ tốt nghiệp.*  
> - *Vì vậy, theo tiêu chuẩn Backdoor của Judea Pearl, tập $Z$ bắt buộc chỉ gồm các biến đo lường trước khi can thiệp xảy ra: $Z_1$ (Điểm đầu vào), $Z_2$ (Nợ học phí), $Z_3$ (Đóng học phí đúng hạn), $Z_4$ (Học vấn của mẹ), $Z_5$ (Tuổi nhập học)."*

---

## VÒNG 1b: MỤC 1.4–1.5 — VÌ SAO LOẠI 794 SINH VIÊN `ENROLLED` KHỎI Y

### 🎯 Vị trí trên Notebook
Mục 1.4 (thống kê quần thể), bảng luồng loại trừ ở mục 1.5 (4,424 → 3,630), bảng lý do lựa chọn 1.1a (dòng "Y = tốt nghiệp vs bỏ học"), và mục 9.D (kết cục giữ chân trên toàn bộ 4,424 sinh viên).

### ❓ Thầy Thiện chất vấn
> *"Cột `Target` có ba giá trị, sao các em chỉ giữ hai? Bỏ 794 sinh viên như vậy có làm kết quả bị chệch không?"*

### 💡 Ý đồ của Thầy
Kiểm tra định nghĩa kết cục (Slide 54), luồng loại trừ (Slide 55), và việc nhóm có nhận ra nguy cơ chệch chọn mẫu khi loại bớt dữ liệu hay không.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, cột `Target` gồm Graduate (2,209), Dropout (1,421) và Enrolled (794). Sinh viên Enrolled là những bạn **vẫn đang học** tại thời điểm thu thập dữ liệu, nên kết cục cuối khóa của họ **chưa xảy ra** — sau này họ có thể tốt nghiệp, cũng có thể bỏ học. Gán họ vào 0 hay 1 đều là đoán, còn giữ thành lớp thứ ba thì $Y$ không còn nhị phân. Vì vậy nhóm loại họ và báo rõ luồng 4,424 → loại 794 → 3,630 theo yêu cầu Slide 55.*
>
> *Nhóm cũng nhận thấy việc loại này **có thể gây chệch**, vì nhóm Enrolled không chia đều: khoảng 20% sinh viên không học bổng còn đang học, so với khoảng 12% ở nhóm có học bổng. Để kiểm tra, ở mục 9.D nhóm dùng một kết cục khác — **giữ chân**, bằng 1 nếu sinh viên tốt nghiệp hoặc vẫn đang học, bằng 0 nếu bỏ học — trên **toàn bộ 4,424 sinh viên**, ước lượng bằng AIPW. Kết quả ATE = **+16.87%**, CI 95% [13.79%, 19.73%]. Hiệu ứng vẫn dương và rõ, nên kết luận 'học bổng giúp sinh viên ở lại học' không phụ thuộc vào việc loại nhóm Enrolled. Con số nhỏ hơn +24% một phần vì định nghĩa kết cục khác nhau."*

---

## VÒNG 1c: MỤC 1.6 — HÌNH 1.2 (MUTUAL INFORMATION): VÌ SAO XANH DÙNG, ĐỎ, XÁM KHÔNG DÙNG

### 🎯 Vị trí trên Notebook
Cell 13 (Hình 1.2) và phần đọc kết quả ở mục 1.6c (Cell 15).

### ❓ Thầy Thiện chất vấn
> *"Hình 1.2 tô xanh các biến được dùng, xám các biến không dùng. Các em chọn biến theo chỉ số MI này à? Vậy sao `Application_mode`, `Course` có MI cao hơn điểm vào và Debtor mà lại không đưa vào? Còn học phí có MI cao nhất thì sao lại tô đỏ?"*

### 💡 Ý đồ của Thầy
Kiểm tra xem nhóm có phân biệt được **chọn biến theo liên kết** (tư duy ML) và **chọn biến theo tiêu chuẩn cửa sau** (tư duy nhân quả) hay không, và nhóm có dám nhận các biến bị bỏ là nhiễu ẩn tiềm năng không.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, **màu sắc không do MI quyết định**. Nhóm chọn $\mathbf{Z}$ trước, theo tiêu chuẩn cửa sau, với hai điều kiện: biến đo **trước hoặc lúc** nhận học bổng, và có lý do ảnh hưởng tới **cả** $T$ lẫn $Y$. Hình 1.2 vẽ sau, chỉ để kiểm tra lại lựa chọn đó.*
>
> *Lý do không chọn theo MI: MI chỉ đo **liên kết**, tức là biết $X$ thì giảm được bao nhiêu bất định về $Y$. Nó không phân biệt $X$ gây ra $Y$, $Y$ gây ra $X$, hay cả hai có chung nguyên nhân. Nếu chọn theo MI, các biến tín chỉ HK1/HK2 sẽ đứng đầu, nhưng chúng là trung gian xảy ra sau học bổng và điều chỉnh theo chúng sẽ làm mất hiệu ứng. Vì vậy chúng đã bị loại khỏi hình ngay từ đầu (Bảng 1.6a).*
>
> *Đọc hình theo ba màu:*
> - ***Xanh** (học bổng, tuổi, điểm vào, Debtor): các biến đã chọn đều thực sự liên kết với kết cục, không có biến 'chết'. Để dễ hình dung: độ bất định ban đầu của $Y$ khoảng 0.67 nats (tỷ lệ tốt nghiệp 60.85%), học bổng với MI 0.066 tương đương khoảng 10%, Debtor với 0.018 khoảng 3%.*
> - ***Đỏ** (học phí đúng hạn, MI 0.101, cao hơn cả học bổng): biến này **có** nằm trong $\mathbf{Z}$ nhưng được tô riêng vì nó đáng ngờ nhất. Thời điểm đo mơ hồ, và sinh viên sắp bỏ học thường ngừng đóng học phí, nên một phần liên kết có thể là **chiều ngược** $Y \to Z$. Nhóm ghi nó là *giả định* trong bảng 1.5, coi $T_2$ chỉ là phân tích khám phá, và nêu ở phần hạn chế.*
> - ***Xám**: `Course` (17 ngành) và `Application_mode` (18 phương thức) có thể ảnh hưởng cả khả năng nhận học bổng lẫn khả năng tốt nghiệp, nên nhóm **thừa nhận** chúng là ứng viên nhiễu ẩn. Nhóm không đưa vào vì chúng có quá nhiều mức: thêm vào mạng Bayes rời rạc sẽ làm bảng CPD của $Y$ phình ra hàng nghìn ô, phần lớn gần như không có dữ liệu, và overlap sẽ vỡ. `Previous_qualification_grade` thì tương quan khá mạnh với điểm vào ($r \approx 0.58$), nên phần lớn thông tin đã có trong $Z$ điểm vào.*
>
> *Để bù cho việc bỏ các biến xám, nhóm dùng phân tích độ nhạy Cinelli–Hazlett ở mục 9.H: $RV_{q=1} = 20.97\%$. Nghĩa là một nhiễu ẩn như ngành học phải giải thích được hơn khoảng 21% phần dư của **cả** $T$ lẫn $Y$ thì mới xóa được hiệu ứng."*

### ⚠️ Lưu ý khi bị hỏi sâu
- `mutual_info_classif` ở đây coi mọi cột là liên tục (ước lượng bằng k-NN), kể cả cột định danh như `Course`. Vì vậy chỉ nên dùng MI để **xếp hạng tương đối**, không đọc như giá trị chính xác.
- Không nói "học phí không được dùng". Biến này **có** trong $\mathbf{Z}$ (là $Z_3$ ở mục 8); màu đỏ chỉ đánh dấu nó là giả định yếu nhất.

---

## VÒNG 2: MỤC 2.3 — NGHỊCH LÝ QUAN SÁT THÔ P(Y|T) VS CAN THIỆP DO(T)

### 🎯 Vị trí trên Notebook
Mục 2.3: `def tinh_toan_do_chenh_lech_quan_sat_tho(...)`.

### ❓ Thầy Thiện chất vấn
> *"Ở mục 2.3, các em in ra màn hình:*  
> - *Xác suất tốt nghiệp khi có học bổng: $P(Y=1 \mid T=1) = 86.17\%$ ($835 / 969$)*  
> - *Xác suất tốt nghiệp khi không có học bổng: $P(Y=1 \mid T=0) = 51.63\%$ ($1,374 / 2,661$)*  
> - *Độ chênh lệch quan sát thô (Naive Associational Gap): $+34.54\%$.*  
> *Vậy nhà trường có thể kết luận rằng cứ cấp học bổng cho sinh viên là tỷ lệ tốt nghiệp sẽ tăng vọt thêm gần $35\%$ được không? Tại sao?"*

### 💡 Ý đồ của Thầy
Kiểm tra Tiêu chí 3 Rubric: Phân biệt giữa quan sát $P(Y \mid T)$ và can thiệp $P(Y \mid \text{do}(T))$.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, **hoàn toàn KHÔNG THỂ kết luận như vậy**! Con số $+34.54\%$ là một tương quan bị thiên lệch nặng nề (Confounded Association).*  
> 
> *Lý do:*  
> - *Trong dữ liệu quan sát, sinh viên nhận học bổng vốn dĩ đã có điểm đầu vào cao hơn ($Z_2$) và tình trạng tài chính gia đình ổn định hơn ($Z_4$). Bản thân những yếu tố nền tảng này đã giúp họ có xác suất tốt nghiệp cao hơn người khác, dù có hay không có học bổng.*  
> - *Nói cách khác, tồn tại các con đường cửa sau (Backdoor paths) $T \leftarrow Z \rightarrow Y$ bơm tương quan giả vào mối quan hệ giữa $T$ và $Y$.*  
> - *Để biết hiệu ứng can thiệp thực sự của nhà trường khi ban hành chính sách cấp học bổng cho một sinh viên bất kỳ, ta bắt buộc phải dùng toán tử phẫu thuật đồ thị $\text{do}(T=1)$ nhằm cắt đứt mọi mũi tên đi vào $T$, cô lập duy nhất dòng chảy nhân quả $T \rightarrow Y$."*

---

## VÒNG 2b: MỤC 2 — HÌNH 2.1: Ô ĐỎ "T=1, CHƯA ĐÓNG HỌC PHÍ" CHỈ CÓ 37 SINH VIÊN

### 🎯 Vị trí trên Notebook
Cell 18 (Hình 2.1, biểu đồ bên phải: tỷ lệ tốt nghiệp theo học bổng × học phí đúng hạn), liên hệ với mục 8.2b (Cell 59–60) và cột `Overlap_Status` của Table B.

### ❓ Thầy Thiện chất vấn
> *"Cột đỏ (có học bổng nhưng chưa đóng học phí đúng hạn) ghi 24.3% với n = 37. Con số này tính thế nào, có tin được không, và nó nói lên điều gì?"*

### 💡 Ý đồ của Thầy
Kiểm tra hiểu biết về **độ bất định của ước lượng trên ô nhỏ** và **positivity** (Slide 55: kiểm tra số người nhận can thiệp trong từng nhóm hiệp biến).

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, trước hết em xin đọc nhãn: 'Học phí đúng hạn = 0' nghĩa là sinh viên **chưa đóng học phí đúng hạn** (khác với biến `Debtor`, mà notebook gọi là 'nợ học phí'). Bốn cột là tỷ lệ tốt nghiệp quan sát trong bốn nhóm:*
>
> | Học bổng | Học phí | Số SV | Tốt nghiệp |
> | :--- | :--- | ---: | ---: |
> | T=0 | chưa đóng | 449 | 4.5% |
> | T=0 | đúng hạn | 2,212 | 61.2% |
> | T=1 | chưa đóng | **37** | **24.3%** (9/37) |
> | T=1 | đúng hạn | 932 | 88.6% |
>
> *Con số 24.3% chỉ là 9 trên 37 sinh viên tốt nghiệp. Với cỡ mẫu nhỏ như vậy, chỉ một sinh viên thay đổi kết cục cũng làm tỷ lệ lệch 2.7 điểm %, và khoảng tin cậy 95% (Wilson) rộng từ khoảng **13% đến 40%**. So với ô 932 sinh viên, khoảng tin cậy chỉ khoảng ±2 điểm. Vì vậy em chỉ đọc con số này là 'thấp, khoảng một phần tư', không dùng nó như một ước lượng chính xác.*
>
> *Ô này mang hai ý nghĩa:*
> - ***Về nội dung:** người chưa đóng học phí đúng hạn gần như không tốt nghiệp; trong 486 sinh viên này chỉ có 29 người tốt nghiệp. Liên kết gần như tất định như vậy củng cố nghi ngờ nhân quả ngược: bỏ học dẫn tới ngừng đóng phí.*
> - ***Về phương pháp:** đây là chỗ **positivity yếu**. Muốn ước lượng $P(Y \mid do(T=1))$ cho một sinh viên chưa đóng học phí đúng hạn, mô hình chỉ có 37 người để học. Khi chia tiếp theo điểm vào, tuổi, Debtor và học vấn mẹ, 37 người này rải ra nhiều ô chỉ còn 0–2 người. Ở những ô đó, CPD của $B_0$ chủ yếu do **làm mịn BDeu** quyết định, tức là dựa vào tiên nghiệm nhiều hơn dữ liệu. Vì vậy nhóm tô đỏ ô này, cảnh báo ở mục 8.2b, đánh dấu hồ sơ thiếu hỗ trợ ở cột `Overlap_Status` của Table B, và chỉ coi $T_2$ (học phí) là phân tích khám phá."*

### ⚠️ Lưu ý
- Đây là số **quan sát, chưa điều chỉnh nhiễu**. Không được đọc 24.3% − 4.5% ≈ +20 điểm thành hiệu ứng nhân quả của học bổng trong nhóm chưa đóng học phí đúng hạn.
- Nếu bị hỏi cách tính khoảng tin cậy: Wilson cho tỷ lệ nhị thức, $n = 37$, $\hat p = 0.243$, $z = 1.96$ ⇒ khoảng [13.4%, 40.1%].

---

## VÒNG 3: B0, H1, H2 — HỌC CẤU TRÚC, MẶT NẠ M VÀ ĐỊNH DANH

### 🎯 Vị trí trên Notebook
Notebook chính: mục 3.3 (HillClimb minh họa), mục 8.1, Hình 8.1 và bảng tóm tắt đồ thị (mục 8.2c), bảng $q_m(t, C_i)$.  
Mã nguồn: notebook chính, mục 8.B (B0), 8.C (H1), 8.D (H2).

### ❓ Thầy Thiện chất vấn
> 1. *"B0 của các em học DAG bằng gì, trên tập nào, chọn thiết lập ra sao? Tại sao lại ép cạnh $T \to Y$?"*  
> 2. *"H2 'neural' ở chỗ nào? Điều kiện để NOTEARS cho ra DAG là gì, các em kiểm tra hội tụ thế nào?"*  
> 3. *"Mặt nạ $M$ để làm gì? Bỏ $M$ đi thì sao?"*

### 💡 Ý đồ của Thầy
Kiểm tra Slide 57–63: B0 học cấu trúc truyền thống có ràng buộc, H1 thay CPD trên đúng DAG B0, H2 học cấu trúc bằng mạng nơ-ron với cùng ràng buộc, chọn thiết lập trên VALIDATION.

### 🗣️ Lời thoại đối đáp chuẩn
> *1. **B0:** `pgmpy.HillClimbSearch` trên **TRAIN** của từng seed, với `ExpertKnowledge` khai báo trước khi học: cấm $Y \to \{T, Z\}$ và $T \to Z$ (trật tự thời gian), bắt buộc $T \to Y$. Cạnh $T \to Y$ là **giả thuyết cần ước lượng**: nếu không có cạnh này thì theo định nghĩa ATE của mô hình bằng 0, nên nhóm khai báo nó là một giả định, không phải một phát hiện. Hàm điểm BIC / BDeu / K2 được so bằng **log-likelihood VALIDATION**; K2 thắng ở cả 5 seed. DAG được đóng băng trước khi chạm TEST. DAG học được có $\text{Pa}(Y) = \{T, \text{Debtor}, \text{Học phí}, \text{Tuổi}\}$ — thưa hơn Domain DAG 11 cạnh (Domain DAG chỉ còn là tham chiếu).*  
>
> *2. **H1:** dùng **đúng DAG B0 của run đó**, chỉ thay CPD của $Y$ bằng MLP nhận đúng $\text{Pa}(Y)$ (4 đầu vào, 769 tham số), loss BCE, Adam, dừng sớm trên VALIDATION; checkpoint lưu ở `artifacts/checkpoints/h1_seed<seed>.pt`. Bảng $q_m(t, C_i)$ trong notebook nạp lại chính checkpoint này.*  
>
> *3. **H2 — NOTEARS-MLP phi tuyến (Zheng et al., 2020):** mỗi nút có một MLP riêng; trọng số cạnh $W_{ij}$ là chuẩn của trọng số tầng 1 đi từ biến $i$ vào MLP của nút $j$. Ràng buộc DAG: $h(W) = \text{tr}(e^{W \circ W}) - d = 0$, giải bằng Augmented Lagrangian; $h(W) = 0$ khi và chỉ khi không có chu trình vì $\text{tr}((W\circ W)^k)$ cộng dồn trọng số các chu trình độ dài $k$. Nhóm **kiểm tra hội tụ**: dừng khi $h < 10^{-6}$; cả 5 seed đạt $h \approx 10^{-7}$. $\lambda_1$ chọn trên VALIDATION seed 42 ($10^{-4}$), ngưỡng $\tau$ chọn trên VALIDATION từng seed (0.1–0.2). Cạnh $T \to Y$ ở H2 **được học**, không bị ép.*  
>
> *4. **Mặt nạ $M$:** dùng đúng danh sách cạnh cấm của B0. $M$ nhân vào trọng số tầng 1 nên cạnh cấm có gradient 0; code `assert` mọi trọng số bị cấm bằng đúng 0 (cột `H2_max|W|_forbidden` = 0 ở mọi seed). Nếu bỏ ràng buộc, mục 3.3 cho thấy ngay cả HillClimb cũng học ra cạnh phi lý `Target_Graduate → Admission_grade` — kết cục không thể gây ra điểm đầu vào.*  
>
> *5. **Định danh:** DoWhy `identify_effect` chạy trên **chính DAG học được** của B0 và H2 ở mọi seed (cột `DoWhy_Identification` trong Table B). Trọng số cạnh chỉ là đại lượng học nội bộ, không phải hiệu ứng nhân quả (Slide 58)."*

---

## VÒNG 4: MỤC 4 — BAYESIAN SMOOTHING (BDEU) VS MAXIMUM LIKELIHOOD (MLE)

### 🎯 Vị trí trên Notebook
Mục 4.2–4.3: `BayesianEstimator` với BDeu prior.

### ❓ Thầy Thiện chất vấn
> *"Khi học CPD cho mô hình B0, tại sao các em không dùng MLE mà lại dùng tiên nghiệm BDeu?"*

### 💡 Ý đồ của Thầy
Kiểm tra Tiêu chí 2 Rubric (Slide 57) và nghiên cứu bóc tách **Ablation A3** (Slide 67).

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, lý do nằm ở **vấn đề bùng nổ không gian điều kiện và ô xác suất bằng 0 (Sparse Cells / Zero-frequency problem)**:*  
> 
> - *Với Domain DAG đầy đủ, $Y$ có 6 cha ($3 \times 2^5 = 96$ cấu hình). DAG B0 học được thưa hơn: $\text{Pa}(Y) = \{T, Z_2, Z_3, Z_5\}$ ($2^4 = 16$ cấu hình), còn H2 có thể thêm $Z_1$ (48 cấu hình). Càng nhiều cấu hình thì càng nhiều ô thưa.*  
> - *Trong tập dữ liệu huấn luyện, có những tổ hợp thực tế cực kỳ hiếm gặp (ví dụ: sinh viên lớn tuổi, nợ học phí, mẹ học vấn thấp nhưng lại được cấp học bổng). Khi đó, số lượng mẫu rơi vào ô này bằng 0 ($N_{c} = 0$).*  
> - *Nếu dùng **MLE thuần túy (Ablation A3)**, thuật toán sẽ gặp lỗi chia cho 0 hoặc gán xác suất cứng bằng 0 tuyệt đối, khiến mô hình bị tê liệt khi suy diễn ngoài mẫu.*  
> - *Nhóm sử dụng **Bayesian smoothing với tiên nghiệm BDeu (Bayesian Dirichlet equivalent uniform)** và cỡ mẫu tương đương $ESS = 5$ cho mô hình $B_0$ dùng trong Tables A/B/H3 (`b0_ess = 5` trong `Config` ở mục 8.A của notebook chính); Cell 11 của notebook chính minh họa thêm với $ESS = 10$. Bản chất của BDeu là phân bổ giả mẫu (pseudo-counts) $\alpha_{ijk} = \frac{ESS}{q_Y \times r_Y}$ vào mọi ô, với $q_Y$ là số cấu hình cha và $r_Y = 2$: ví dụ $q_Y = 16$ (B0 học được) cho $\frac{5}{32} \approx 0.16$ giả mẫu mỗi ô; $q_Y = 96$ (Domain DAG) cho $\frac{5}{192} \approx 0.026$. Độ nhạy theo ESS được quét trong `tools/run_ablations_and_extensions.py`. Mức ESS nhỏ này cung cấp một lực tiên nghiệm (prior strength) vừa đủ mạnh để xóa sạch các ô xác suất 0, bảo đảm tính ổn định số học vững chắc khi suy diễn can thiệp, nhưng đồng thời không làm méo mó các phân phối thực nghiệm đã quan sát từ $3,630$ mẫu."*

---

## VÒNG 5: MỤC 7 — KHUNG LÀM VIỆC DOWHY 4 BƯỚC & 2 BÀI TEST BÁC BỎ

### 🎯 Vị trí trên Notebook
Mục 7: `CausalModel` của DoWhy.

### ❓ Thầy Thiện chất vấn
> *"Ở mục 7, các em dùng thư viện DoWhy để kiểm định nhân quả:*  
> 1. *Quy trình 4 bước chuẩn của DoWhy được thực hiện trong code như thế nào?*  
> 2. *Các em chạy 2 bài kiểm tra bác bỏ `placebo_treatment_refuter` và `data_subset_refuter`. Ý nghĩa của từng bài kiểm tra là gì và làm sao biết nó 'Pass' hay 'Fail'?"*

### 💡 Ý đồ của Thầy
Kiểm tra Tiêu chí 8 Rubric (Slide 70, 75): Tính vững bác bỏ của ước lượng nhân quả.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy:*  
> 
> *1. **Về quy trình 4 bước của DoWhy:** Nhóm tuân thủ đúng 4 hàm chuẩn:*  
> - **Bước 1: Model (`CausalModel`)** — Khai báo dữ liệu, biến treatment $T$, outcome $Y$, và truyền đồ thị Primary DAG vào.  
> - **Bước 2: Identify (`identify_effect`)** — DoWhy dùng thuật toán phân tích đồ thị để tìm công thức định danh (xác nhận Backdoor criterion thỏa mãn với tập $Z$).  
> - **Bước 3: Estimate (`estimate_effect`)** — Ước lượng hiệu ứng can thiệp ATE.  
> - **Bước 4: Refute (`refute_estimate`)** — Chạy các bài kiểm tra giả định để tìm cách bác bỏ kết quả vừa tìm được.  
> 
> *2. **Về 2 bài kiểm tra bác bỏ (Refuters):**  
> - **Bài 1: `placebo_treatment_refuter` (Can thiệp giả dược):** Code sẽ tráo đổi ngẫu nhiên nhãn học bổng $T$ giữa các sinh viên thành một biến giả ngẫu nhiên. Nếu mô hình thực sự phản ánh nhân quả, hiệu ứng của biến giả dược này bắt buộc phải triệt tiêu về 0. Mục 7 (DoWhy trên Domain DAG, `random_state=42`, tái lập được) cho: $+0.2412 \to +0.0047 \approx 0$; Table C (DAG B0 học được) cho $+0.0016$. Cả hai đều **Pass**.  
> - **Bài 2: `data_subset_refuter` (Độ nhạy trên tập con):** Code sẽ lấy mẫu ngẫu nhiên $80\%$ dữ liệu và ước lượng lại ATE. Nếu hiệu ứng nhân quả là ổn định, ATE trên tập con không được phép biến động mạnh. Mục 7: $+0.2412 \to +0.2447$; Table C (DAG B0): $+0.2309 \to +0.2265$. Hiệu ứng ổn định, **Pass**. Lưu ý: các refuter chỉ kiểm tra độ ổn định của bộ ước lượng, **không** chứng minh giả định không có nhiễu ẩn — phần đó do $RV$ Cinelli–Hazlett (Cell 28) đảm nhận."*

---

## VÒNG 6: BẢNG A, B, C — SO SÁNH B0 / H1 / H2 VÀ QUY TẮC TÍNH ATE

### 🎯 Vị trí trên Notebook
Mục 8.2 (hiển thị Tables A–C), Hình 8.3 (ATE theo seed), bảng $q_m(t, C_i)$, mục 8.3.  
Mã nguồn sinh bảng: notebook chính, mục 8.E–8.F (chạy trực tiếp, ≈ 3–4 phút).

### ❓ Thầy Thiện chất vấn
> 1. *"Neural CPD hay học cấu trúc nơ-ron có giúp dự báo tốt hơn B0 không? Như vậy có nghĩa là ước lượng nhân quả tốt hơn không?"*  
> 2. *"Chỉ cho thầy cách tính ATE và chứng minh không dùng $Y_{\text{test}}$. CI 95% tính bằng gì?"*  
> 3. *"Refuter trong Table C kiểm tra cái gì? Có kiểm tra được H1 không?"*

### 🗣️ Lời thoại đối đáp chuẩn
> *1. **Dự báo — null finding:** AUROC $B_0 = 0.7996 \pm 0.0119$, $H_1 = 0.8012 \pm 0.0130$, $H_2 = 0.8034 \pm 0.0155$. Hiệu ghép cặp $\Delta_{H_1 - B_0} = +0.0016 \pm 0.0025$, $\Delta_{H_2 - B_0} = +0.0038 \pm 0.0038$ — **không lớn hơn độ tản giữa các run**; Brier gần như không đổi ($0.1659 / 0.1656 / 0.1666$). Nhóm kết luận **không có bằng chứng** phần nơ-ron cải thiện dự báo. Macro-F1 của B0 và H1 trùng nhau vì $\text{Pa}(Y)$ chỉ có 16 cấu hình nên hai mô hình gán nhãn giống nhau tại ngưỡng chọn trên VALIDATION. Và dù dự báo có tốt hơn thì cũng **không** chứng minh ATE đúng hơn (Slide 48).*  
>
> *2. **ATE (Slide 59, 61):** với mỗi hồ sơ $C_i$ trong $C_{\text{test}}$ ($n = 545$), $q_m(t, C_i) = P_m(Y=1 \mid T=t, \text{Pa}(Y)\setminus T = C_i)$ — hợp lệ vì mọi biến trong $\text{Pa}(Y)\setminus T$ là tiền can thiệp (mặt nạ cấm $T \to Z$). $\text{risk}_m(t) = \frac{1}{n}\sum_i q_m(t, C_i)$, $\text{ATE}_m = \text{risk}_m(1) - \text{risk}_m(0)$. Hàm `query_model_effects` chỉ đọc các cột cha của $Y$ và gán $T$ thành 0/1; cột $Y$ của test chỉ dùng để chấm dự báo (Table A). Kết quả: $B_0 = +0.2418 \pm 0.0046$, $H_1 = +0.2374 \pm 0.0114$, $H_2 = +0.2358 \pm 0.0067$ — kết luận can thiệp **không đổi** giữa ba mô hình. **CI 95%:** percentile bootstrap 1000 lần trên các hồ sơ $C_{\text{test}}$, **giữ cố định mô hình** — chỉ phản ánh bất định lấy mẫu quần thể tham chiếu ($\approx \pm 0.008$), không gồm bất định tham số hay đồ thị; bất định giữa các run báo riêng bằng mean ± std (Slide 68).*  
>
> *3. **Refuter (Slide 70–71):** DoWhy placebo + subset chạy trên **DAG của B0** và **DAG của H2** với ước lượng hỗ trợ `backdoor.linear_regression` (placebo $+0.0016$; subset $+0.2265$ và $+0.2233$). Chúng kiểm tra *đồ thị + ước lượng tuyến tính*, không kiểm tra CPD BDeu. Với H1 nơ-ron, refuter tuyến tính không áp dụng được, nên nhóm cài một **kiểm định huấn luyện lại** có ghi nhãn riêng: hoán vị $T$ rồi huấn luyện lại MLP → ATE $+0.0177 \approx 0$; huấn luyện lại trên 80% TRAIN → $+0.2365$ (gốc $+0.2453$). Vượt qua refuter **không** chứng minh DAG đúng hay không có nhiễu ẩn.*  
>
> *4. **Chi phí:** truy vấn dùng numpy advanced indexing trên mảng CPD (≈ 0.2 ms cho 545 hồ sơ); thời gian khớp trung bình mỗi run trên CPU: B0 ≈ 0.4 s, H1 ≈ 1.5–2 s, H2 ≈ 14–20 s tùy máy và lần chạy (Table C ghi rõ phần cứng). Chi tiết epoch và quá trình huấn luyện: Vòng 6b."*

---

## VÒNG 6b: MỤC 8.E2 — QUÁ TRÌNH HUẤN LUYỆN, EPOCH, OVERFIT VÀ EARLY STOPPING

### 🎯 Vị trí trên Notebook
Cell 53–54 (mục 8.E2): Hình 8.2 (loss của H1 theo epoch, hội tụ của H2), Bảng 8.E2 (AUROC / Accuracy / log-loss trên TRAIN, VAL, TEST), Hình 8.2b và Bảng 8.E2b (H1 có và không có early stopping). Cấu hình nằm ở Cell 41, vòng huấn luyện H1 ở Cell 47, H2 ở Cell 49.

### ❓ Thầy Thiện chất vấn
> 1. *"Mô hình nơ-ron của các em train bao nhiêu epoch? Có nhanh không? Validation trong lúc train làm thế nào, lưu ở đâu?"*
> 2. *"Không thấy vẽ loss hay accuracy, sao biết mô hình có overfit không? Còn tập TEST dùng ở đâu?"*
> 3. *"Early stopping có thật sự cần không? Nếu bỏ đi thì sao?"*

### 💡 Ý đồ của Thầy
Kiểm tra quy trình huấn luyện có đúng chuẩn TRAIN / VALIDATION / TEST (Slide 56, 68), và nhóm có thực sự hiểu mô hình nơ-ron của mình hay chỉ gọi thư viện.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, nhóm em trả lời lần lượt:*
>
> *1. **Epoch và thời gian.** $B_0$ không có epoch, vì CPD BDeu chỉ là đếm tần suất có làm mịn, khớp một lần mất khoảng 0.4 giây. $H_1$ là MLP 769 tham số, huấn luyện tối đa **150 epoch** bằng Adam (lr = 0.005, batch 64), có **early stopping với patience 15**: sau mỗi epoch, nhóm tính loss trên VALIDATION; nếu 15 epoch liên tiếp không cải thiện thì dừng và lấy lại trọng số tốt nhất. Thực tế các seed dừng sau 17–39 epoch, epoch tốt nhất là 2–4 (riêng seed 43 là 24), mất khoảng 1.5–2 giây. $H_2$ không dùng epoch mà dùng augmented Lagrangian: mỗi vòng ngoài 300 bước Adam, tối đa 12 vòng, dừng khi $h(W) < 10^{-6}$; cả 5 seed hội tụ sau 7–8 vòng, mất khoảng 14–20 giây. Toàn bộ chạy trên CPU.*
>
> *Lịch sử loss được lưu trong `run_artifacts[seed]['H1_log']` (`train_loss`, `val_loss`) và `['H2_res']` (`h_history`, `recon_history`), đồng thời ghi vào checkpoint `artifacts/checkpoints/h1_seed*.pt`. Có thể mở `notebook/training_log_viewer.ipynb` để xem log từng epoch kiểu Keras của cả hai bản H1 (có và không early stopping).*
>
> *2. **Overfit và vai trò của TEST.** Hình 8.2 vẽ loss TRAIN và VAL của $H_1$ qua từng epoch, và đường hội tụ $h(W)$ của $H_2$. Bảng 8.E2 tính AUROC, accuracy và log-loss trên **cả ba tập**. AUROC của $B_0$ là 0.802 / 0.800 / 0.800 trên TRAIN / VAL / TEST, của $H_1$ là 0.801 / 0.800 / 0.801. Chênh TRAIN − TEST chỉ +0.002 và ≈ 0, nên không overfit. $H_2$ chênh lớn hơn một chút (+0.008), vì nó thêm điểm vào làm cha của $Y$, nhưng vẫn nhỏ hơn độ dao động giữa các seed (±0.016). TEST chỉ dùng **một lần ở cuối**, sau khi mọi lựa chọn đã đóng băng trên VALIDATION; bảng này chỉ đọc lại mô hình, không chỉnh gì theo TEST, và AUROC TEST trùng với Table A.*
>
> *3. **Có và không có early stopping.** Nhóm huấn luyện thêm một bản $H_1$ chạy đủ 150 epoch, giữ trọng số epoch cuối, và lưu thành checkpoint riêng (`h1_seed<s>_no_early_stop.pt`, bên cạnh `h1_seed<s>_early_stop.pt`). Trung bình 5 seed, bản không dừng sớm có loss TRAIN thấp hơn (0.498 so với 0.500) nhưng loss VAL cao hơn (0.504 so với 0.500), log-loss TEST cao hơn (0.501 so với 0.499), AUROC TEST thấp hơn (0.799 so với 0.801). Đó đúng là dấu hiệu bắt đầu overfit. Mức chênh nhỏ vì đầu vào chỉ có 16 tổ hợp. ATE trung bình gần như không đổi (+0.236 so với +0.237), nhưng từng seed lệch tới ±0.02, nên early stopping chủ yếu giúp ước lượng ổn định hơn."*

### ⚠️ Lưu ý
- **Vì sao hội tụ sớm?** Cha của $Y$ trong DAG $B_0$ là $\{T, Z_2, Z_3, Z_5\}$, đều nhị phân, nên đầu vào chỉ có $2^4 = 16$ tổ hợp. MLP thực chất chỉ cần học 16 xác suất, vài epoch là đủ. Đây cũng là lý do $H_1$ không tốt hơn $B_0$ (null finding RQ2): bảng CPD đã biểu diễn được hết.
- **Loss TRAIN cao hơn VAL ở vài seed:** loss TRAIN được đo khi dropout 0.2 đang bật; loss VAL đo ở chế độ đánh giá, không dropout.
- **Sai số tái tạo của $H_2$ tăng dần** qua các vòng là bình thường: phạt chu trình càng chặt thì mô hình càng phải hy sinh độ khớp để là một DAG hợp lệ.
- Không nói "mô hình không overfit vì AUROC cao". Bằng chứng là **khoảng cách giữa TRAIN và TEST** nhỏ, và đường loss VAL không đi xuống thêm khi huấn luyện lâu hơn.

---

## VÒNG 7: MỤC 8.2 & BẢNG MULTI-TREATMENT — MỞ RỘNG CAN THIỆP (T1, T2, T3) VÀ CHẨN ĐOÁN H2

### 🎯 Vị trí trên Notebook
Mục 8.2 (bảng Multi-Treatment), bảng positivity $T_1 \times T_2$ và mục 8.2b.  
Script thực nghiệm: `tools/verify_and_compute_multi_treatments.py`.

### ❓ Thầy Thiện chất vấn
> 1. *"Bản chất của $T_2$ và $T_3$ là gì? Tại sao $T_2$ đạt $ATE \approx +50\%$, gấp hơn 2 lần học bổng?"*  
> 2. *"Can thiệp kép $T_3$ có hiệu ứng cộng hưởng không? Cột 'AIPW' trên test split có thật là doubly robust không?"*  
> 3. *"Vì sao H2 ở Table B giống hệt nhau qua 5 seed và CI có độ rộng bằng 0?"*

### 💡 Ý đồ của Thầy
Kiểm tra xem nhóm có phân biệt được **biến can thiệp hợp lệ** với **biến hậu quả/trạng thái**, có hiểu điều kiện positivity, và có trung thực về bộ ước lượng đã cài đặt hay không.

### 🗣️ Lời thoại đối đáp chuẩn
> *"Dạ thưa Thầy, nhóm trình bày $T_1$ là kết quả chính, còn $T_2, T_3$ chỉ là **phân tích khám phá có hạn chế**:*  
>
> *1. **$T_1$ (Học bổng) — kết quả chính:** ATE $\approx +21.8\%$ (LPM), $+21.8\%$ (G-computation), $+23.1\%$ (Outcome Regression) trên $C_{\text{test}}$; p-value lớn nhất qua 5 split cỡ $10^{-21}$. Nhất quán với Table B ($B_0$: $+23.63\%$, $H_1$: $+23.09\%$).*  
>
> *2. **$T_2$ (`Tuition fees up to date`) — KHÔNG phải chính sách xóa nợ:** đây là **trạng thái đóng học phí ghi trong hồ sơ**. Bảng positivity ở mục 8.2b cho thấy trong 449 sinh viên không học bổng và không đóng phí đúng hạn, tỷ lệ tốt nghiệp chỉ $4.5\%$. Nhiều khả năng **bỏ học dẫn đến ngừng đóng phí** (nhân quả ngược), nên con số $+45\% \sim +51\%$ là **liên kết đã điều chỉnh**, không phải hiệu ứng của một chính sách. Nhóm không đưa nó vào khuyến nghị.*  
>
> *3. **$T_3$ (tương phản kép):** $do(T_1{=}1,T_2{=}1)$ vs $do(T_1{=}0,T_2{=}0)$ cho $+67\% \sim +75\%$, nhưng xấp xỉ **tổng** $T_1 + T_2$ (LPM: $0.218 + 0.451 \approx 0.667$) ⇒ **không có bằng chứng cộng hưởng**. Ô $T_1{=}1, T_2{=}0$ chỉ có **37 sinh viên** (≈5 SV mỗi test split) ⇒ positivity yếu, ước lượng phụ thuộc ngoại suy mô hình.*  
>
> *4. **Về bộ ước lượng:** AIPW đúng nghĩa (có số hạng hiệu chỉnh $\frac{T(Y-\hat\mu_1)}{\hat e}$) **chỉ chạy trên Full Cohort**. Trên $C_{\text{test}}$, do nguyên tắc không dùng $Y_{\text{test}}$, bộ ước lượng thứ ba là **Outcome Regression (T-learner)**: $\frac{1}{n}\sum_i[\hat\mu_1(Z_i) - \hat\mu_0(Z_i)]$ với $\hat\mu_t$ học trên train — bảng đã ghi đúng tên này. CI lấy từ bootstrap refit trên train ($B=100$).*  
>
> *5. **Về H2 ở Table B:** H2 dùng NOTEARS-MLP đã hội tụ, học được $\text{Pa}(Y) \supseteq \{T, Z_2, Z_3, Z_5\}$ (thêm $Z_1$ ở 4/5 seed), cho ATE $+0.2358 \pm 0.0067$, sát B0. Vì $Z_1$ có 3 trạng thái, 1–2% hồ sơ test của H2 không có đủ hỗ trợ ở cả $T=0$ và $T=1$ trong TRAIN — được báo ở cột `Overlap_Status` của Table B (Slide 79)."*

---

## VÒNG 8: MỤC 9–10 — CATE, PHÂN BỔ HỌC BỔNG (+164 SV) & CHUYỂN DỊCH SANG KAGGLE

### 🎯 Vị trí trên Notebook
Mục 9.F (CATE), mục 9.4, mục 10 (Table 10.1, 10.2) và mục 10.3.

### ❓ Thầy Thiện chất vấn
> 1. *"Bảng CATE ghi nhóm nợ học phí 'hưởng lợi gấp 2.03 lần'. Các em ước lượng CATE bằng gì?"*  
> 2. *"Con số +164 cử nhân 'mỗi năm, chi phí 0 đồng' được đánh giá như thế nào? Có bị lạc quan không?"*  
> 3. *"Kết quả trên Kaggle có chứng minh cơ chế nhân quả bất biến không?"*

### 🗣️ Lời thoại đối đáp chuẩn
> *1. **CATE:** Nhóm ước lượng theo **nhóm con định trước** bằng G-computation logistic khớp riêng trong từng nhóm, không phải Causal Forest. Cột "1.72x / 2.03x" là **tỷ số rủi ro** $\text{Risk}(do(1))/\text{Risk}(do(0))$. Xét **hiệu ứng tuyệt đối**, nhóm nợ học phí ($+20.27\%$) thậm chí không cao hơn nhóm không nợ ($+21.35\%$); tỷ số lớn chỉ vì mức nền thấp ($19.74\%$). Nhóm lớn tuổi có CATE $+28.02\%$ so với $+16.08\%$ ở nhóm đúng tuổi. Các nhóm chưa có khoảng tin cậy, nên nhóm chỉ coi đây là gợi ý.*  
>
> *2. **+164 SV:** Đây là con số **mô phỏng in-sample** trên toàn bộ 3,630 hồ sơ (gộp nhiều khóa, không phải mỗi năm). Cùng một cặp mô hình $\hat\mu_0, \hat\mu_1$ vừa dùng để xếp hạng vừa dùng để chấm điểm chính sách, nên con số có xu hướng lạc quan. Danh sách xếp hạng còn dùng $Z_5$ (trạng thái học phí), biến có nguy cơ là hệ quả của bỏ học. Muốn khẳng định cần cross-fitting và đánh giá off-policy (IPW/DR), hoặc tốt nhất là thí điểm có kiểm soát. Vì vậy ở bảng tổng kết, RQ4 được ghi là **"Ủng hộ có điều kiện"**.*  
>
> *3. **Kaggle:** Kết quả **trái chiều**. Mô hình học trên UCI cho ATE gần nhau trên UCI ($0.2153$) và trên Kaggle ($0.2143$), nhưng mô hình học trên Kaggle cho $0.30$–$0.32$. Ngoài ra Kaggle S4E6 là dữ liệu **tổng hợp** sinh từ UCI, nên đây không phải bằng chứng về tính bất biến trên một quần thể thực khác."*

---

## BẢNG TỔNG HỢP CÁC CON SỐ CỐT LÕI CẦN THUỘC LÒNG KHI BẢO VỆ

| Hạng mục | Con số thực tế trong Notebook | Ý nghĩa giải thích |
| :--- | :---: | :--- |
| **Cỡ mẫu sau lọc** | **3,630 sinh viên** | Đã loại 794 sinh viên `Enrolled` để nhị phân hóa biến $Y$ (`Graduate` vs `Dropout`). |
| **Tỷ lệ nhận học bổng ($T$)** | **$26.7\%$** ($969 / 3,630$) | Tỷ lệ can thiệp thực tế trong tập dữ liệu UCI ID 697. |
| **Chênh lệch quan sát thô** | **$+34.54\%$** ($86.17\% - 51.63\%$) | Tương quan thuần túy từ mục 2.3 ($835/969$ vs $1,374/2,661$, $N=3,630$), bị thổi phồng do biến nhiễu $\mathbf{Z}$ (chưa can thiệp). |
| **MI với $Y$ (Hình 1.2)** | Học phí $0.101$ > học bổng $0.066$ > tuổi $0.061$ | Chỉ là liên kết; màu do tiêu chuẩn cửa sau quyết định, không do MI. |
| **Ô thưa Hình 2.1** | $T=1$, chưa đóng học phí đúng hạn: **$9/37 = 24.3\%$**, CI 95% ≈ [13%, 40%] | Positivity yếu; CPD ở đây chủ yếu do làm mịn BDeu. |
| **Huấn luyện H1 / H2** | H1: tối đa 150 epoch, patience 15, dừng sau 17–39 epoch; H2: 7–8 vòng ngoài, $h(W) \approx 10^{-7}$ | Đầu vào H1 chỉ 16 tổ hợp nhị phân ⇒ hội tụ sớm. |
| **Overfit (Bảng 8.E2)** | AUROC TRAIN − TEST: B0 $+0.002$, H1 $\approx 0$, H2 $+0.008$ | Không overfit đáng kể; bỏ early stopping làm log-loss VAL/TEST tăng nhẹ. |
| **AUROC (Table A)** | **B0 $0.7996 \pm 0.0119$ / H1 $0.8012 \pm 0.0130$ / H2 $0.8034 \pm 0.0155$** | Hiệu ghép cặp nhỏ hơn độ tản giữa run ⇒ null finding cho phần nơ-ron. |
| **ATE (Table B)** | **B0 $+0.2418 \pm 0.0046$ / H1 $+0.2374 \pm 0.0114$ / H2 $+0.2358 \pm 0.0067$** | CI 95% bootstrap trên $C_{\text{test}}$, mô hình cố định (≈ ±0.008 mỗi seed). |
| **DAG học được** | B0: HillClimb + K2 (chọn trên VAL), $\text{Pa}(Y)=\{T,Z_2,Z_3,Z_5\}$; H2: NOTEARS-MLP, $h(W)\approx 10^{-7}$, $\tau$ = 0.1–0.2 | Cạnh $T \to Y$: bắt buộc ở B0 (giả thuyết), được học ở H2. |
| **$T_2$: Trạng thái học phí đúng hạn** | **$+45\% \sim +51\%$** (chỉ là liên kết) | KHÔNG phải chính sách xóa nợ; nguy cơ nhân quả ngược (bỏ học → ngừng đóng phí). |
| **$T_3$: Tương phản kép** | **$+67\% \sim +75\%$** | ≈ $T_1 + T_2$ (không cộng hưởng); positivity yếu: ô $T_1{=}1,T_2{=}0$ chỉ 37 SV. |
| **Độ nhạy Cinelli–Hazlett** | **$RV_{q=1} = 20.97\%$**, $RV_{\alpha=0.05} = 18.37\%$ | Tính trực tiếp ở Table 5.3b; bản cũ ghi $23.40\%$ là do dùng sai công thức. |
| **Kiểm định DoWhy Refuters** | **Mục 7: Placebo +0.0047, Subset +0.2447**<br/>**Table C (DAG B0): Placebo +0.0016, Subset +0.2265** | Đều Pass: ước lượng ổn định; không thay thế cho phân tích nhiễu ẩn. |
| **Phân bổ theo CATE** | **+164 SV** (65.37% vs 60.85%) | Mô phỏng in-sample trên 3,630 hồ sơ, chưa cross-fit ⇒ ủng hộ có điều kiện. |
| **Kaggle S4E6** | UCI→UCI $0.2153$, UCI→Kaggle $0.2143$, Kaggle→UCI $0.3217$ | Trái chiều; Kaggle là dữ liệu tổng hợp ⇒ không chứng minh bất biến. |

---
*Kịch bản này đã được lưu vĩnh viễn tại [`report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md). Bạn có thể mở file ra ôn luyện bất cứ lúc nào trước buổi bảo vệ!*
