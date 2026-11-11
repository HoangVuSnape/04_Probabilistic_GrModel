# Công thức dễ hiểu

Mỗi công thức gồm bốn phần: **công thức**, **bảng ký hiệu**, **ý nghĩa dễ hiểu** và **ví dụ đơn giản**. Số liệu trong ví dụ lấy từ dự án; ví dụ nào là số giả định để minh họa thì có ghi rõ *(giả định)*.

Ký hiệu chung của dự án: $T$ = có học bổng (1) / không (0); $Y$ = tốt nghiệp (1) / bỏ học (0); $Z$ = các biến nền đo lúc nhập học (điểm vào, Debtor, học phí đúng hạn, học vấn mẹ, tuổi).

---

## Phần A — Câu hỏi nhân quả

### A1. Chênh lệch quan sát thô

$$\text{Gap} = P(Y=1 \mid T=1) - P(Y=1 \mid T=0)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $P(Y=1 \mid T=1)$ | Tỷ lệ tốt nghiệp **trong nhóm đang có** học bổng |
| $P(Y=1 \mid T=0)$ | Tỷ lệ tốt nghiệp trong nhóm không có học bổng |
| dấu "$\mid$" | "biết rằng", "trong số những người có…" |

**Ý nghĩa:** so sánh thẳng hai nhóm như trong dữ liệu. Con số này **trộn** hiệu ứng thật của học bổng với sự khác biệt sẵn có giữa hai nhóm (người có học bổng vốn điểm cao hơn, ít nợ hơn).

**Ví dụ:** $86.17\% - 51.63\% = +34.54$ điểm %. Sau khi điều chỉnh nhiễu chỉ còn khoảng $+24$, tức khoảng 10 điểm là do hai nhóm vốn đã khác nhau.

### A2. Toán tử can thiệp $do(\cdot)$

$$P(Y=1 \mid do(T=1)) \quad \text{khác với} \quad P(Y=1 \mid T=1)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $P(Y \mid T=1)$ | **Quan sát**: nhìn những người *đang* có học bổng |
| $P(Y \mid do(T=1))$ | **Can thiệp**: nếu nhà trường *chủ động cấp* học bổng cho mọi người |

**Ý nghĩa:** $do(T=1)$ là "bắt buộc đặt" $T=1$, cắt mọi lý do khiến một người vốn có hay không có học bổng.

**Ví dụ đời thường:** người mang ô thường gặp đường ướt (quan sát). Nhưng nếu *ép* mọi người mang ô ($do$), đường không ướt thêm — vì trời mưa mới là nguyên nhân chung.

### A3. Hiệu ứng nhân quả trung bình (ATE)

$$\text{ATE} = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\mathbb{E}[\cdot]$ | Giá trị trung bình (kỳ vọng); với $Y$ là 0/1 thì chính là tỷ lệ |
| $\mathbb{E}[Y \mid do(T=1)]$ | Tỷ lệ tốt nghiệp nếu **ai cũng** được cấp học bổng |
| $\mathbb{E}[Y \mid do(T=0)]$ | Tỷ lệ tốt nghiệp nếu **không ai** được cấp |

**Ý nghĩa:** học bổng làm tỷ lệ tốt nghiệp của cả quần thể thay đổi bao nhiêu điểm %.

**Ví dụ:** B0 seed 42: $0.7863 - 0.5397 = +0.2467$ → cấp học bổng cho mọi người thì tỷ lệ tốt nghiệp tăng khoảng 24.7 điểm %.

### A4. Điều kiện dương (positivity / overlap)

$$0 < P(T=1 \mid Z=z) < 1 \quad \text{với mọi } z$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $P(T=1 \mid Z=z)$ | Xác suất nhận học bổng của những người có hồ sơ nền $z$ |
| $z$ | Một hồ sơ cụ thể, ví dụ "điểm cao, không nợ, tuổi ≤ 20" |

**Ý nghĩa:** với **mỗi** loại hồ sơ, phải có cả người được và không được học bổng thì mới so sánh được. Nếu một nhóm toàn người có học bổng, mô hình phải "đoán mò".

**Ví dụ:** ô "có học bổng nhưng chưa đóng đúng hạn" chỉ có 37 SV → positivity yếu, ước lượng ở đó kém tin cậy.

---

## Phần B — Đồ thị và mạng Bayes

### B1. Phân rã mạng Bayes

$$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P\big(X_i \mid \text{Pa}(X_i)\big)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $X_1, \dots, X_n$ | Các biến (ở đây 7 nút: $Z_1..Z_5, T, Y$) |
| $\text{Pa}(X_i)$ | **Cha** của $X_i$: các nút có mũi tên đi vào $X_i$ |
| $\prod$ | Nhân tất cả lại |
| $P(X_i \mid \text{Pa}(X_i))$ | Bảng CPD của nút $X_i$ |

**Ý nghĩa:** thay vì một bảng khổng lồ cho mọi tổ hợp 7 biến, chỉ cần mỗi nút một bảng nhỏ "nếu biết cha thì nút này ra sao".

**Ví dụ:** đồ thị $Z \to T \to Y$ và $Z \to Y$: $P(z,t,y) = P(z)\,P(t \mid z)\,P(y \mid t,z)$.

### B2. Phân rã cắt cụt (graph surgery)

$$P(\mathbf{z}, m, y \mid do(T=t)) = \prod_k P(z_k)\; P(m \mid t)\; P(y \mid t, m, \mathbf{z})$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\mathbf{z}$ | Các biến nền |
| $m$ | Biến trung gian (tín chỉ HK1) |
| thừa số **bị bỏ** | $P(t \mid \mathbf{z})$ — cách học bổng "vốn được trao" |

**Ý nghĩa:** can thiệp = **cắt mọi mũi tên đi vào $T$** rồi đặt $T=t$. Trong công thức, chỉ cần bỏ đúng một thừa số $P(t \mid \mathbf{z})$.

**Ví dụ:** Hình 6.1(b): các mũi tên đỏ nét đứt đi vào "Học bổng" bị cắt; mũi tên từ biến nền sang "Tốt nghiệp" giữ nguyên.

### B3. Công thức điều chỉnh cửa sau (backdoor adjustment)

$$P(Y=1 \mid do(T=t)) = \sum_{z} P(Y=1 \mid T=t, Z=z)\; P(Z=z)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\sum_z$ | Cộng qua mọi loại hồ sơ nền $z$ |
| $P(Y=1 \mid T=t, Z=z)$ | Tỷ lệ tốt nghiệp **trong cùng loại hồ sơ** $z$, khi $T=t$ |
| $P(Z=z)$ | Tỷ lệ người thuộc loại hồ sơ $z$ trong quần thể |

**Ý nghĩa:** so sánh có học bổng / không học bổng **trong từng nhóm giống nhau**, rồi lấy trung bình có trọng số theo cỡ nhóm. Như vậy hai bên được so "cùng điều kiện".

**Ví dụ *(giả định)*:** 60% SV điểm thấp, 40% điểm cao.
Điểm thấp: tốt nghiệp 50% (không HB) / 70% (có HB). Điểm cao: 70% / 85%.
$P(Y \mid do(1)) = 0.6 \times 0.70 + 0.4 \times 0.85 = 0.76$; $P(Y \mid do(0)) = 0.6 \times 0.50 + 0.4 \times 0.70 = 0.58$ → ATE $= +0.18$.

### B4. Truy vấn của B0/H1/H2 trên tập kiểm thử

$$q_m(t, C_i) = P_m\big(Y=1 \mid T=t,\ \text{Pa}(Y)\setminus T = C_i\big),\qquad \text{risk}_m(t) = \frac{1}{n}\sum_{i=1}^{n} q_m(t, C_i),\qquad \text{ATE}_m = \text{risk}_m(1) - \text{risk}_m(0)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $m$ | Mô hình: B0, H1 hoặc H2 |
| $C_i$ | Hồ sơ nền của sinh viên thứ $i$ trong tập TEST ($n = 545$) |
| $\text{Pa}(Y)\setminus T$ | Các cha của $Y$ trừ $T$ (ví dụ Debtor, học phí, tuổi) |
| $q_m(t, C_i)$ | Xác suất tốt nghiệp của sinh viên $i$ **nếu** đặt $T=t$ |
| $\text{risk}_m(t)$ | Trung bình trên 545 hồ sơ |

**Ý nghĩa:** chính là công thức B3, nhưng lấy trung bình trên **các hồ sơ thật của tập TEST** thay vì cộng qua mọi tổ hợp. Không dùng $Y_\text{test}$.

**Ví dụ:** một hồ sơ (không nợ, đóng đúng hạn, ≤ 20 tuổi): $q_{B0}(0) = 0.7364$, $q_{B0}(1) = 0.9134$ → riêng người này hơn 17.7 điểm. Trung bình 545 hồ sơ: $0.7863 - 0.5397 = +0.2467$.

---

## Phần C — Máy "học" từ dữ liệu như thế nào

**Phần này trả lời một câu hỏi duy nhất: mô hình lấy các con số và các mũi tên từ đâu ra?**

Hình dung mạng Bayes như một **sơ đồ + bộ bảng tra**:
- **Sơ đồ** (đồ thị): cái gì ảnh hưởng cái gì — ví dụ "Debtor → Tốt nghiệp".
- **Bảng tra** (CPD): với mỗi tình huống, xác suất là bao nhiêu — ví dụ "nếu đang nợ và có học bổng thì 60% tốt nghiệp".

Máy phải học **cả hai** từ dữ liệu:

| Việc cần học | Cách làm trong dự án | Mục |
| :--- | :--- | :--- |
| Các con số trong bảng tra | Đếm tần suất (MLE), có "làm mịn" cho ô ít dữ liệu (BDeu) | C1, C2 |
| Sơ đồ mũi tên của B0 | Thử thêm/bớt mũi tên, chấm điểm từng sơ đồ (BIC), chọn sơ đồ đoán tốt nhất trên dữ liệu chưa thấy (VAL) | C3, C4 |
| Sơ đồ mũi tên của H2 | Mạng nơ-ron tự học độ mạnh mũi tên, có luật cấm vòng tròn | C5, C6 |

---

### C1. Đếm tần suất (MLE)

**Câu hỏi:** trong bảng tra, ô "đang nợ + có học bổng" nên ghi bao nhiêu % tốt nghiệp?

**Hình dung:** muốn biết tỷ lệ đỗ của lớp, đếm số người đỗ chia cho sĩ số. Không có gì hơn.

**Ví dụ từng bước *(giả định)*:**
1. Lọc ra những SV "đang nợ + có học bổng": có **50** người.
2. Đếm trong 50 người đó ai tốt nghiệp: **30** người.
3. Chia: $30 / 50 = 60\%$. Ghi 60% vào ô.

**Công thức:**
$$\hat P(Y=1 \mid \text{tình huống}) = \frac{\text{số người tốt nghiệp trong tình huống đó}}{\text{số người thuộc tình huống đó}} = \frac{N(y,\ \text{pa})}{N(\text{pa})}$$

**Đọc bằng lời:** "pa" là viết tắt của *parents* — tức tình huống của các biến cha (nợ hay không, học bổng hay không…). $N(\cdot)$ là "đếm số người".

**Điểm yếu:** ô có ít người thì con số rất ẩu. Ví dụ chỉ có 4 người và cả 4 tốt nghiệp → máy ghi **100%**, như thể chắc chắn — trong khi chỉ là may.

---

### C2. Đếm có "làm mịn" (BDeu) — cách dự án thật sự dùng

**Câu hỏi:** làm sao để ô ít người không cho ra 0% hay 100% vô lý?

**Hình dung:** chấm điểm món ăn mới chỉ có 2 lượt đánh giá 5 sao. Bạn không tin nó "hoàn hảo"; bạn ngầm cộng thêm vài lượt đánh giá "trung bình" cho chắc. Món có 1,000 lượt đánh giá thì vài lượt cộng thêm chẳng ảnh hưởng gì.

BDeu làm y như vậy: **cộng thêm một ít "người ảo" chia đều cho tốt nghiệp và bỏ học** vào mỗi ô trước khi chia.

**Ví dụ từng bước *(giả định)*:**
1. Ô hiếm: 4 người, cả 4 tốt nghiệp. Đếm thô: $4/4 = 100\%$.
2. Cộng người ảo: mỗi bên (tốt nghiệp / bỏ học) được thêm khoảng **0.16** người.
3. Tính lại: $\dfrac{4 + 0.16}{4 + 0.16 + 0.16} = \dfrac{4.16}{4.32} \approx 96\%$.
4. Ô đông (ví dụ 900/1000): $\dfrac{900.16}{1000.32} \approx 90\%$ — gần như không đổi.

**Công thức:**
$$\hat P(Y=y \mid \text{pa}) = \frac{N(y,\ \text{pa}) + \frac{\alpha}{r\,q}}{N(\text{pa}) + \frac{\alpha}{q}}$$

**Đọc bằng lời:**
- $\alpha = 5$: **tổng số người ảo** cho cả bảng của $Y$ (gọi là ESS).
- $q$: số tình huống (ô) trong bảng — người ảo được chia đều cho các ô.
- $r = 2$: số kết cục (tốt nghiệp / bỏ học) — trong mỗi ô lại chia đôi.
- Vậy mỗi bên của mỗi ô nhận $\alpha / (r\,q) = 5/(2 \times 16) \approx 0.16$ người ảo.

**Vì sao dự án dùng BDeu:** có những tình huống rất hiếm (ví dụ chỉ 37 SV "có học bổng + chưa đóng học phí"); không làm mịn thì xác suất ở đó nhảy lung tung.

---

### C3. Chấm điểm một sơ đồ (BIC)

**Câu hỏi:** giữa nhiều sơ đồ mũi tên khác nhau, sơ đồ nào "tốt"?

**Hình dung:** chấm bài thi có hai phần: **điểm nội dung** (giải thích dữ liệu tốt không) **trừ điểm dài dòng** (dùng quá nhiều mũi tên). Một bài vẽ mũi tên lung tung có thể "khớp" dữ liệu hơn chút xíu nhưng bị trừ nặng vì rườm rà.

**Ví dụ từng bước *(giả định)*:**
1. Sơ đồ A: điểm khớp $-2000$, có 30 ô trong các bảng.
2. Sơ đồ B = A thêm 1 mũi tên: điểm khớp tốt hơn **3 điểm** ($-1997$) nhưng thêm **4 ô**.
3. Mức trừ cho mỗi ô thêm $= \tfrac12 \log(2541) \approx 3.9$ điểm → 4 ô bị trừ $\approx 15.7$.
4. Lợi 3, mất 15.7 → **không thêm** mũi tên đó.

**Công thức:**
$$\text{BIC} = \underbrace{\log L}_{\text{điểm khớp dữ liệu}} - \underbrace{\frac{k}{2}\log N}_{\text{trừ điểm rườm rà}}$$

**Đọc bằng lời:** $\log L$ = đồ thị giải thích dữ liệu tốt cỡ nào (luôn là số âm, càng gần 0 càng tốt); $k$ = số ô trong các bảng tra; $N$ = số sinh viên trong TRAIN (2,541).

**Trong dự án:** HillClimb thử lần lượt "thêm / bớt / đảo một mũi tên", giữ thay đổi nào làm điểm tăng. Dự án thử 3 kiểu chấm điểm (BIC, BDeu, K2) — giống 3 giám khảo có tiêu chí hơi khác nhau.

---

### C4. Chọn giám khảo nào? Thi lại trên đề mới (log-likelihood VAL)

**Câu hỏi:** 3 kiểu chấm điểm cho ra 3 sơ đồ khác nhau. Tin sơ đồ nào?

**Hình dung:** ba học sinh ôn thi theo ba cách. Muốn biết cách nào tốt thật, cho cả ba làm **một đề mới chưa từng thấy** — ai làm tốt hơn thì cách ôn đó tốt hơn. Tập VALIDATION chính là "đề mới".

**Ví dụ từng bước:**
1. Mỗi sơ đồ (từ BIC, BDeu, K2) được học bảng tra trên TRAIN.
2. Đưa 544 SV của tập VAL vào: với mỗi người, hỏi sơ đồ "khả năng thấy đúng hồ sơ này là bao nhiêu?".
3. Lấy log của các khả năng đó rồi cộng lại thành một điểm (số âm; **càng gần 0 càng tốt**).
4. Seed 42: K2 được $-2208.37$, BIC được $-2211.11$ → **K2 thắng**. K2 thắng ở cả 5 seed.

**Công thức:**
$$\text{LL}_\text{VAL} = \sum_{\text{mỗi SV trong VAL}}\ \sum_{\text{mỗi biến}} \log P(\text{giá trị thật} \mid \text{giá trị của các biến cha})$$

**Vì sao dùng log:** nhân rất nhiều xác suất nhỏ sẽ ra số tí hon; lấy log biến phép nhân thành phép cộng, dễ tính và dễ so sánh.

---

### C5. Luật cấm vòng tròn của H2 — $h(W)$

**Câu hỏi:** H2 dùng mạng nơ-ron tự học mũi tên. Làm sao bắt nó **không** vẽ vòng tròn kiểu "A gây ra B, B lại gây ra A"?

**Hình dung:**
- $W$ là bảng 7×7 ghi **độ mạnh** mỗi mũi tên (0 = không có mũi tên).
- $h(W)$ là **"máy dò vòng tròn"**: nó đi theo mọi đường mũi tên, cứ đường nào **quay về đúng chỗ xuất phát** thì cộng điểm.
  - Không có vòng nào → máy dò báo **0**.
  - Có vòng → báo **số dương**; vòng càng mạnh, số càng lớn.

**Ví dụ từng bước:**
1. Chỉ có A → B (độ mạnh 0.5): đi từ A sang B rồi hết đường, không quay về A → $h = 0$.
2. Thêm B → A (0.4): giờ A → B → A quay về chỗ cũ → máy dò báo $h \approx 0.04$.
3. Mạng nơ-ron bị **phạt** theo $h$ → nó buộc phải làm yếu một trong hai mũi tên đến khi $h$ về 0.

**Công thức (chỉ cần biết nó là "máy dò vòng tròn"):**
$$h(W) = \operatorname{tr}\big(e^{W \circ W}\big) - d$$

**Đọc bằng lời từng mảnh:**
- $W \circ W$: bình phương mọi độ mạnh cho dương hết (để các số không triệt tiêu nhau).
- $e^{(\cdot)}$: cộng dồn **mọi đường đi** dài 1 bước, 2 bước, 3 bước…
- $\operatorname{tr}$ (vết): chỉ giữ lại những đường đi **quay về chính nó**.
- $- d$: trừ đi phần "đứng yên tại chỗ" luôn có sẵn (bằng số biến, 7), để không vòng nào thì ra đúng 0.

**Vì sao phải dùng công thức này thay vì đếm vòng bình thường:** mạng nơ-ron học bằng cách "nhích dần" (gradient), nên cần một con số thay đổi **mượt** khi mũi tên mạnh lên hay yếu đi. Đếm vòng thông thường chỉ trả lời có/không, không nhích dần được.

**Trong dự án:** cả 5 seed đưa $h$ về khoảng $0.0000001$ ($10^{-7}$) — coi như không còn vòng.

---

### C6. Ba mong muốn cùng lúc của H2 (hàm mục tiêu)

**Câu hỏi:** khi học, H2 cố tối ưu cái gì?

**Hình dung:** như tìm nhà với 3 tiêu chí cùng lúc: **gần chỗ làm** (khớp dữ liệu), **giá rẻ** (ít mũi tên), và **bắt buộc có giấy tờ hợp lệ** (không vòng tròn). Hai tiêu chí đầu được đánh đổi; tiêu chí thứ ba thì càng về sau càng siết chặt, đến khi bắt buộc.

**Công thức:**
$$\text{Tổng phạt} = \underbrace{\text{sai số dự đoán}}_{\text{khớp dữ liệu}} + \underbrace{\lambda_1 \times \text{tổng độ mạnh mũi tên}}_{\text{ít mũi tên}} + \underbrace{\alpha\, h(W) + \tfrac{\rho}{2}\, h(W)^2}_{\text{phạt vòng tròn, tăng dần}}$$

**Đọc bằng lời:**
- **Sai số dự đoán:** mỗi biến được đoán lại từ các biến khác; đoán càng trật càng phạt.
- $\lambda_1$ (= 0.0001, chọn trên VAL): mức phạt cho mỗi mũi tên, để sơ đồ gọn.
- $\alpha, \rho$: mức phạt vòng tròn. Mỗi "vòng ngoài" (300 bước học) mà vòng tròn chưa hết, **phạt tăng gấp 10**.

**Ví dụ (seed 42):**

| Vòng ngoài | $h(W)$ (độ "vòng tròn") | Sai số dự đoán |
| :---: | :---: | :---: |
| 1 | 0.032 | 0.436 |
| 4 | 0.00015 | 0.437 |
| 7 | 0.00000033 → **hội tụ** | 0.492 |

Vòng tròn biến mất dần; đổi lại sai số dự đoán tăng một chút — đó là cái giá của việc bắt sơ đồ phải hợp lệ.

---

> **Tóm lại Phần C trong 3 câu:**
> 1. Bảng tra = **đếm tần suất**, cộng thêm chút "người ảo" để ô hiếm không ra 0% / 100% (BDeu).
> 2. Sơ đồ của B0 = **thử thêm/bớt mũi tên**, chấm điểm "khớp trừ rườm rà" (BIC/K2…), rồi chọn bằng **thi lại trên tập VAL**.
> 3. Sơ đồ của H2 = **mạng nơ-ron học độ mạnh mũi tên**, có "máy dò vòng tròn" $h(W)$ ép về 0.

---

## Phần D — Mạng nơ-ron H1

### D1. Hàm sigmoid

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $z$ | Đầu ra thô của mạng (logit), từ $-\infty$ đến $+\infty$ |
| $p$ | Xác suất tốt nghiệp, trong khoảng 0–1 |

**Ý nghĩa:** "nén" một số bất kỳ thành xác suất.

**Ví dụ:** $z = 0 \Rightarrow p = 0.5$; $z = 2 \Rightarrow p = 0.88$; $z = -2 \Rightarrow p = 0.12$.

### D2. Hàm mất mát BCE (binary cross-entropy)

$$\text{BCE} = -\frac{1}{n}\sum_{i}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $y_i$ | Nhãn thật (1 tốt nghiệp / 0 bỏ học) |
| $p_i$ | Xác suất mô hình dự đoán |

**Ý nghĩa:** phạt dự đoán sai, phạt **rất nặng** khi sai mà tự tin. Đây là thứ được tối thiểu hóa khi huấn luyện, và là "loss" trong Hình 8.2.

**Ví dụ:** SV tốt nghiệp ($y=1$): đoán $p=0.8$ → loss $= -\log 0.8 = 0.22$; đoán $p=0.2$ → loss $= -\log 0.2 = 1.61$.

### D3. Early stopping

$$\text{dừng khi } \text{BCE}_\text{val} \text{ không giảm trong } 15 \text{ epoch liên tiếp; giữ } \theta^* = \arg\min_{\text{epoch}} \text{BCE}_\text{val}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| epoch | Một lượt đi hết tập TRAIN |
| patience = 15 | Số epoch được phép "không tiến bộ" |
| $\theta^*$ | Trọng số ở epoch tốt nhất trên VAL |

**Ý nghĩa:** dừng trước khi mô hình bắt đầu học thuộc nhiễu.

**Ví dụ:** seed 42: tốt nhất ở epoch 3, chạy thêm 15 epoch không khá hơn → dừng ở epoch 18, lấy lại trọng số epoch 3.

---

## Phần E — Đánh giá dự báo

### E1. AUROC

$$\text{AUROC} = P\big(p_\text{tốt nghiệp} > p_\text{bỏ học}\big)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $p_\text{tốt nghiệp}$ | Xác suất mô hình gán cho một SV tốt nghiệp ngẫu nhiên |
| $p_\text{bỏ học}$ | … cho một SV bỏ học ngẫu nhiên |

**Ý nghĩa:** khả năng **xếp hạng** đúng: 0.5 = đoán bừa, 1.0 = hoàn hảo.

**Ví dụ:** AUROC $\approx 0.80$ → bốc ngẫu nhiên một SV tốt nghiệp và một SV bỏ học, 80% số lần mô hình cho người tốt nghiệp điểm cao hơn.

### E2. Brier score

$$\text{Brier} = \frac{1}{n}\sum_i (p_i - y_i)^2$$

**Ý nghĩa:** sai số bình phương của xác suất; càng nhỏ càng tốt (0 = hoàn hảo).

**Ví dụ:** $p=0.8, y=1$: $(0.8-1)^2 = 0.04$. Dự án: Brier $\approx 0.166$.

### E3. Macro-F1

$$F_1 = \frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}},\qquad \text{Macro-}F_1 = \tfrac{1}{2}\big(F_1^{(\text{tốt nghiệp})} + F_1^{(\text{bỏ học})}\big)$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| Precision | Trong số đoán "tốt nghiệp", bao nhiêu % đúng |
| Recall | Trong số thật sự tốt nghiệp, bắt được bao nhiêu % |

**Ý nghĩa:** cân bằng giữa "đoán đúng" và "không bỏ sót", tính đều cho cả hai lớp. Ngưỡng cắt được chọn trên VAL.

**Ví dụ *(giả định)*:** Precision 0.8, Recall 0.75 → $F_1 = 2 \times 0.6 / 1.55 = 0.77$.

### E4. Khoảng tin cậy bootstrap (percentile)

$$\text{CI}_{95\%} = \big[\text{phân vị } 2.5\%,\ \text{phân vị } 97.5\%\big] \text{ của } \Big\{\overline{d}^{(b)}\Big\}_{b=1}^{1000},\quad d_i = q(1,C_i) - q(0,C_i)$$

**Ý nghĩa:** lấy mẫu lại 545 hồ sơ (có hoàn lại) 1000 lần, mỗi lần tính ATE; 95% giá trị ở giữa là khoảng tin cậy. Mô hình giữ cố định.

**Ví dụ:** B0 seed 42: ATE $= 0.2467$, CI $[0.2394, 0.2541]$.

### E5. Khoảng tin cậy Wilson cho một tỷ lệ

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\hat p$ | Tỷ lệ quan sát |
| $n$ | Cỡ mẫu |
| $z$ | 1.96 cho 95% |

**Ý nghĩa:** tỷ lệ tính trên ít người thì khoảng tin cậy rất rộng.

**Ví dụ:** ô 37 SV, $\hat p = 9/37 = 24.3\%$ → CI $\approx [13.4\%, 40.1\%]$.

### E6. Mutual Information

$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}$$

**Ý nghĩa:** biết $X$ thì giảm được bao nhiêu bất định về $Y$. Bằng 0 khi $X$, $Y$ độc lập. Chỉ đo **liên kết**, không đo nhân quả.

**Ví dụ:** học phí đúng hạn có MI 0.101, cao hơn học bổng (0.066) — nhưng đó là dấu hiệu nghi nhân quả ngược, không phải lý do chọn biến.

---

## Phần F — Phân tích mở rộng (mục 9–10)

### F1. Propensity score và AIPW

$$e(x) = P(T=1 \mid X=x)$$
$$\widehat{\text{ATE}}_\text{AIPW} = \frac{1}{n}\sum_i\left[\hat\mu_1(x_i) + \frac{T_i\,(Y_i - \hat\mu_1(x_i))}{e(x_i)}\right] - \frac{1}{n}\sum_i\left[\hat\mu_0(x_i) + \frac{(1-T_i)(Y_i - \hat\mu_0(x_i))}{1-e(x_i)}\right]$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $e(x)$ | Xác suất được học bổng của người có hồ sơ $x$ (propensity) |
| $\hat\mu_1(x), \hat\mu_0(x)$ | Dự đoán khả năng ở lại trường nếu có / không có học bổng |
| phần $\frac{T(Y-\hat\mu_1)}{e}$ | "Sửa sai" phần dự đoán bằng sai số thật, có trọng số |

**Ý nghĩa:** kết hợp hai cách: dự đoán kết cục + cân lại theo xác suất được chọn. Chỉ cần **một trong hai** đúng là kết quả không chệch (doubly robust).

**Ví dụ *(giả định, một SV)*:** có học bổng, ở lại ($Y=1$), $\hat\mu_1 = 0.8$, $e = 0.4$: điểm $= 0.8 + (1-0.8)/0.4 = 1.3$. Trung bình trên 4,424 SV cho ATE giữ chân $= +16.87\%$.

### F2. Trung gian: NDE, NIE

$$M = a\,T + \dots,\qquad Y = \text{NDE}\cdot T + b\,M + \dots,\qquad \text{NIE} = a \times b,\qquad \text{tỷ lệ trung gian} = \frac{\text{NIE}}{\text{NDE} + \text{NIE}}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $M$ | Số tín chỉ đạt HK1 (biến trung gian) |
| $a$ | Học bổng làm tăng bao nhiêu tín chỉ |
| $b$ | Mỗi tín chỉ thêm làm tăng tốt nghiệp bao nhiêu |
| NDE | Tác dụng **trực tiếp** (không qua HK1) |
| NIE | Tác dụng **gián tiếp** qua HK1 |

**Ý nghĩa:** tách xem học bổng giúp "vì học tốt hơn ở HK1" bao nhiêu phần.

**Ví dụ:** $a = 0.729$ tín chỉ, $b \approx 0.064$/tín chỉ → NIE $\approx 0.0465$; NDE $= 0.1696$ → tỷ lệ trung gian $= 0.0465 / 0.2161 = 21.5\%$.

### F3. CATE và tỷ số rủi ro

$$\text{CATE}(g) = \mathbb{E}[Y \mid do(T=1), G=g] - \mathbb{E}[Y \mid do(T=0), G=g],\qquad \text{RR}(g) = \frac{\mathbb{E}[Y \mid do(1), g]}{\mathbb{E}[Y \mid do(0), g]}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $g$ | Một nhóm con (ví dụ "trên 20 tuổi") |
| CATE | Hiệu ứng **tuyệt đối** trong nhóm (điểm %) |
| RR | Hiệu ứng **tương đối** (gấp bao nhiêu lần) |

**Ý nghĩa:** học bổng giúp nhóm nào nhiều hơn.

**Ví dụ:** nhóm > 20 tuổi: $66.74\% - 38.72\% = +28.0$ điểm; RR $= 66.74 / 38.72 = 1.72\times$. "1.72x" là tỷ số, **không phải** "hưởng lợi gấp 1.72 lần nhóm khác".

### F4. Robustness Value (Cinelli & Hazlett)

$$f = \frac{|t|}{\sqrt{df}},\qquad RV = \tfrac{1}{2}\left(\sqrt{f^4 + 4f^2} - f^2\right),\qquad R^2_{Y\sim T\mid Z} = \frac{t^2}{t^2 + df}$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $t$ | Thống kê t của hệ số học bổng trong hồi quy $Y \sim T + Z$ |
| $df$ | Bậc tự do (số mẫu − số hệ số) |
| $RV$ | Một nhiễu ẩn phải giải thích ít nhất bao nhiêu % phương sai dư của **cả** $T$ lẫn $Y$ để xóa hiệu ứng |

**Ý nghĩa:** đo kết luận "khó bị lật" cỡ nào trước một biến chưa quan sát được.

**Ví dụ:** $t = 14.20$, $df = 3623$ → $f = 0.236$ → $RV = 20.97\%$; partial $R^2 = 5.27\%$. Biến quan sát mạnh nhất (học phí) chỉ $10.4\%$ → nhiễu ẩn phải mạnh gấp đôi học phí mới lật được.

### F5. Giá trị của một chính sách phân bổ

$$V(\pi) = \frac{1}{n}\sum_i\Big[\pi_i\,\hat\mu_1(x_i) + (1-\pi_i)\,\hat\mu_0(x_i)\Big]$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\pi_i$ | 1 nếu chính sách cấp học bổng cho SV $i$, 0 nếu không |
| $\hat\mu_1, \hat\mu_0$ | Dự đoán tốt nghiệp nếu có / không có học bổng |

**Ý nghĩa:** ước lượng tỷ lệ tốt nghiệp nếu áp dụng một cách phân bổ; ưu tiên người có $\hat\mu_1 - \hat\mu_0$ (CATE) lớn nhất.

**Ví dụ *(giả định, 3 SV, 1 suất)*:** CATE lần lượt 0.10, 0.30, 0.05 → cấp cho SV 2 lợi nhất. Dự án: cùng 969 suất, tỷ lệ dự kiến $60.85\% \to 65.37\%$ (+164 SV, ước lượng lạc quan vì chưa cross-fit).
