# Glossary — Từ khóa, định nghĩa và công thức

Dự án: **Causal PGM — đánh giá tác động của học bổng lên tốt nghiệp (UCI 697)**, học phần IT701100.
Mỗi mục gồm **định nghĩa**, **công thức** (nếu có, kèm chú thích ký hiệu) và **trong dự án** — nơi khái niệm xuất hiện và giá trị thực tế.
Số liệu lấy từ `artifacts/tables/*.csv` sinh bởi notebook chính `00_MASTER_Causal_PGM_Educational_Intervention.ipynb` (mục 8–10).

**Ký hiệu chung**

| Ký hiệu | Ý nghĩa |
| :--- | :--- |
| $T$ | Can thiệp (treatment): `Scholarship holder`, $T \in \{0, 1\}$ |
| $Y$ | Kết cục (outcome): `Target`, Graduate $= 1$, Dropout $= 0$ |
| $\mathbf{Z} = (Z_1, \dots, Z_5)$ | Biến nền (confounder): điểm đầu vào, Debtor, học phí đúng hạn, học vấn của mẹ, tuổi > 20 |
| $M_1$ | Biến trung gian: số tín chỉ đỗ học kỳ 1 |
| $C_i$ | Hồ sơ hiệp biến nền của sinh viên $i$ trong tập TEST (không gồm $Y_i$) |
| $\text{Pa}(X)$ | Tập nút cha của $X$ trong DAG |
| $n_{\text{test}}$ | Số sinh viên tập TEST ($= 545$ mỗi seed) |
| $m$ | Mô hình: $B_0$, $H_1$ hoặc $H_2$ |

---

## Mục lục
1. [Dữ liệu và thiết kế thực nghiệm](#1-dữ-liệu-và-thiết-kế-thực-nghiệm)
2. [Đồ thị và nền tảng nhân quả](#2-đồ-thị-và-nền-tảng-nhân-quả)
3. [Định danh (identification)](#3-định-danh-identification)
4. [Học cấu trúc và học tham số](#4-học-cấu-trúc-và-học-tham-số)
5. [Ba mô hình B0, H1, H2](#5-ba-mô-hình-b0-h1-h2)
6. [Ước lượng hiệu ứng](#6-ước-lượng-hiệu-ứng)
7. [Bất định và độ ổn định](#7-bất-định-và-độ-ổn-định)
8. [Đánh giá dự báo](#8-đánh-giá-dự-báo)
9. [Kiểm định bác bỏ và độ nhạy](#9-kiểm-định-bác-bỏ-và-độ-nhạy)
10. [Phân tích mở rộng](#10-phân-tích-mở-rộng)
11. [Thư viện và artifact](#11-thư-viện-và-artifact)

---

## 1. Dữ liệu và thiết kế thực nghiệm

### UCI 697
**Định nghĩa.** Bộ dữ liệu *Predict Students' Dropout and Academic Success* (Martins et al., 2021), DOI `10.24432/C5MC89`, giấy phép CC BY 4.0.
**Trong dự án.** 4,424 sinh viên × 37 cột; loại 794 sinh viên `Enrolled` (chưa có kết cục) ⇒ **N = 3,630** (2,209 Graduate, 1,421 Dropout). Không có dòng trùng lặp và không có giá trị thiếu ở các biến mô hình (mục 1.5 notebook chính).

### Eligibility flow (luồng loại trừ)
**Định nghĩa.** Bảng số dòng trước/sau từng bước lọc, để người đọc biết quần thể phân tích được tạo ra thế nào (Slide 55).
**Trong dự án.** 4,424 → loại 794 `Enrolled` → 3,630.

### Variable dictionary (từ điển biến)
**Định nghĩa.** Bảng ghi ý nghĩa, mã hóa, thời điểm đo, vai trò nhân quả và mức thiếu của từng biến (Slide 54).
**Trong dự án.** Mục 1.5 notebook chính; $Z_3$ (học phí đúng hạn) được đánh dấu **thời điểm đo mơ hồ**.

### Train / Validation / Test split
**Định nghĩa.** Chia dữ liệu thành ba phần rời nhau: TRAIN để khớp mô hình, VALIDATION để chọn thiết lập, TEST để đánh giá cuối cùng sau khi đã đóng băng mọi lựa chọn.
**Trong dự án.** 70 / 15 / 15 = 2,541 / 544 / 545 sinh viên; phân tầng theo cặp $(T, Y)$; chỉ số lưu tại `artifacts/splits/split_seed_<seed>.json`.

### Stratified split (chia phân tầng)
**Định nghĩa.** Giữ nguyên tỷ lệ các nhóm (ở đây là 4 ô $(T, Y)$) trong mỗi phần dữ liệu.
**Hệ quả trong dự án.** Bảng đếm $(T, Y)$ của TRAIN và TEST **giống hệt nhau ở cả 5 seed**; vì vậy một mô hình chỉ phụ thuộc $T$ sẽ cho kết quả trùng nhau giữa các seed.

### Paired runs (lượt chạy ghép cặp)
**Định nghĩa.** Trong mỗi seed, mọi mô hình dùng **cùng** hàng TRAIN/VALIDATION/TEST, nên có thể so sánh hiệu từng cặp trong cùng seed.
**Trong dự án.** 5 seed: 42, 43, 44, 45, 46.

### Reference cohort $C_{\text{test}}$ (quần thể tham chiếu)
**Định nghĩa.** Tập hồ sơ hiệp biến nền của sinh viên TEST, dùng làm quần thể chung để chuẩn hóa can thiệp cho mọi mô hình (Slide 56, 61). $Y_{\text{test}}$ **không** dùng để tính hiệu ứng.

### Data leakage (rò rỉ dữ liệu)
**Định nghĩa.** Thông tin của TEST (hoặc của tương lai) lọt vào bước khớp mô hình, làm kết quả đánh giá lạc quan giả tạo.
**Trong dự án.** Mọi tiền xử lý chỉ `fit` trên TRAIN; biến tín chỉ HK1/HK2 (xảy ra sau $T$) không đưa vào $\mathbf{Z}$.

---

## 2. Đồ thị và nền tảng nhân quả

### DAG (Directed Acyclic Graph)
**Định nghĩa.** Đồ thị có hướng không có chu trình; mỗi cạnh $X \to Y$ biểu diễn giả định "$X$ là nguyên nhân trực tiếp của $Y$".

### Causal Bayesian Network (CBN)
**Định nghĩa.** Một DAG kèm phân phối xác suất phân rã theo DAG, và **thêm giả định** rằng can thiệp vào một biến chỉ thay cơ chế của chính biến đó.
**Công thức (factorization).**
$$P(x_1, \dots, x_d) = \prod_{j=1}^{d} P\big(x_j \mid \text{pa}(x_j)\big)$$
- $\text{pa}(x_j)$: giá trị các nút cha của $X_j$.

### CPD (Conditional Probability Distribution)
**Định nghĩa.** Phân phối $P(X_j \mid \text{Pa}(X_j))$ của từng nút. Với biến rời rạc, CPD là một bảng (TabularCPD) có $r_j \times q_j$ ô.
- $r_j$: số trạng thái của $X_j$; $q_j = \prod_{k \in \text{Pa}(j)} r_k$: số cấu hình cha.
**Trong dự án.** DAG B0 có $\text{Pa}(Y) = \{T, Z_2, Z_3, Z_5\}$ ⇒ $q_Y = 2^4 = 16$ cấu hình.

### Domain DAG (đồ thị miền)
**Định nghĩa.** DAG do chuyên gia đề xuất từ kiến thức và trật tự thời gian, trước khi nhìn dữ liệu.
**Trong dự án.** 7 nút, 11 cạnh ($Z_k \to T$, $Z_k \to Y$, $T \to Y$); bản 8 nút thêm $T \to M_1 \to Y$. Chỉ dùng làm **tham chiếu** và cho DoWhy ở mục 7; B0 học DAG riêng.

### Temporal ordering / tiers (trật tự thời gian)
**Định nghĩa.** Xếp biến theo thời điểm xảy ra; cạnh chỉ được đi từ tầng sớm sang tầng muộn.
**Trong dự án.** Tầng 1: $\mathbf{Z}$ (nhập học) → Tầng 2: $T$ → Tầng 3: $M_1$ → Tầng 4: $Y$.

### Confounder (biến gây nhiễu)
**Định nghĩa.** Biến là nguyên nhân chung của $T$ và $Y$ ($T \leftarrow Z \rightarrow Y$), tạo tương quan không nhân quả giữa chúng.

### Mediator (biến trung gian)
**Định nghĩa.** Biến nằm trên đường nhân quả $T \to M \to Y$. Điều chỉnh theo mediator khi ước lượng **tổng** hiệu ứng sẽ chặn mất một phần hiệu ứng (over-control).
**Trong dự án.** $M_1$ (tín chỉ HK1) chỉ dùng cho phân tích trung gian, **không** đưa vào tập điều chỉnh.

### Collider
**Định nghĩa.** Nút có hai mũi tên đi vào trên một đường ($A \rightarrow C \leftarrow B$). Đường bị chặn tự nhiên tại collider; **điều chỉnh** theo collider sẽ mở đường và gây chệch.

### d-separation
**Định nghĩa.** Tiêu chuẩn đọc độc lập có điều kiện từ DAG: $X$ và $Y$ d-tách bởi $S$ nếu mọi đường giữa chúng bị chặn khi biết $S$ (một đường bị chặn nếu chứa non-collider thuộc $S$, hoặc collider mà cả nó lẫn con cháu đều không thuộc $S$).

### Intervention $do(T = t)$ và graph surgery
**Định nghĩa.** Phép can thiệp đặt $T$ bằng $t$ cho mọi đơn vị, tương đương **cắt mọi cạnh đi vào $T$** trong DAG.
**Công thức (truncated factorization).**
$$P\big(\mathbf{z}, m, y \mid do(T = t)\big) = \prod_k P(z_k)\; P(m \mid t)\; P(y \mid t, m, \mathbf{z})$$
**Trong dự án.** Hình 6.1(b) của notebook chính vẽ đồ thị sau phẫu thuật.

### Observational vs interventional
**Định nghĩa.** $P(Y \mid T = t)$ là xác suất **quan sát** (bị nhiễm đường cửa sau); $P(Y \mid do(T = t))$ là xác suất **can thiệp**. Nói chung $P(Y \mid T = t) \neq P(Y \mid do(T = t))$.

### Naive associational gap (chênh lệch thô)
**Công thức.** $\Delta_{\text{naive}} = P(Y = 1 \mid T = 1) - P(Y = 1 \mid T = 0)$
**Trong dự án.** $86.17\% - 51.63\% = \mathbf{+34.54\%}$ (835/969 vs 1,374/2,661).

---

## 3. Định danh (identification)

### Identification
**Định nghĩa.** Chứng minh rằng một đại lượng nhân quả (ví dụ $P(Y \mid do(T))$) **tính được từ phân phối quan sát** dưới DAG và các giả định đã nêu. Khác với *ước lượng* (dùng dữ liệu hữu hạn để tính số) và *khám phá nhân quả* (học DAG từ dữ liệu) — Slide 45.

### Backdoor path (đường cửa sau)
**Định nghĩa.** Đường từ $T$ đến $Y$ bắt đầu bằng một mũi tên **đi vào** $T$ (ví dụ $T \leftarrow Z \rightarrow Y$).

### Backdoor criterion và backdoor adjustment
**Định nghĩa.** Tập $S$ thỏa tiêu chuẩn cửa sau nếu (i) không chứa con cháu của $T$ và (ii) chặn mọi đường cửa sau từ $T$ đến $Y$. Khi đó:
$$P\big(Y = 1 \mid do(T = t)\big) = \sum_{\mathbf{s}} P(Y = 1 \mid T = t, S = \mathbf{s})\; P(S = \mathbf{s})$$
**Trong dự án.** Trên DAG đã học, $\text{Pa}(Y) \setminus \{T\}$ là tập tiền can thiệp (mặt nạ cấm $T \to Z$) nên thỏa tiêu chuẩn cửa sau. B0: $\{Z_2, Z_3, Z_5\}$. DoWhy trả về tập tối thiểu $\{Z_3, Z_5\}$.

### Adjustment set (tập điều chỉnh)
**Định nghĩa.** Tập biến dùng trong công thức cửa sau. Có thể có nhiều tập hợp lệ; tập tối thiểu là tập nhỏ nhất vẫn chặn mọi đường cửa sau.

### Ignorability / No unmeasured confounding
**Định nghĩa.** $Y(t) \perp T \mid \mathbf{Z}$: sau khi biết $\mathbf{Z}$, việc nhận học bổng độc lập với kết cục tiềm năng. **Không kiểm chứng được từ dữ liệu**; đánh giá bằng phân tích độ nhạy (mục 9).

### Consistency
**Định nghĩa.** Nếu $T_i = t$ thì $Y_i = Y_i(t)$: kết cục quan sát bằng kết cục tiềm năng ứng với can thiệp đã nhận. Đòi hỏi "học bổng" là một can thiệp được định nghĩa rõ (một phiên bản).

### Positivity / Overlap
**Định nghĩa.** $0 < P(T = 1 \mid \mathbf{Z} = \mathbf{z}) < 1$ với mọi $\mathbf{z}$ có xác suất dương: mỗi nhóm hiệp biến đều có cả người nhận và không nhận học bổng.
**Trong dự án.** Kiểm tra theo từng hồ sơ test (cột `Overlap_Status` của Table B): B0, H1 đủ hỗ trợ; H2 có **1–2%** hồ sơ test thiếu hỗ trợ.

### Unsupported profile (hồ sơ không được hỗ trợ)
**Định nghĩa.** Hồ sơ $C_i$ mà trong TRAIN không có đồng thời người $T = 0$ và người $T = 1$ mang cùng cấu hình tập điều chỉnh. Ước lượng ở đó dựa vào làm mịn hoặc ngoại suy (Slide 79).

### Reverse causation (nhân quả ngược)
**Định nghĩa.** Tương quan sinh ra do $Y$ (hoặc một hệ quả của $Y$) ảnh hưởng ngược lên biến được coi là nguyên nhân.
**Trong dự án.** `Tuition fees up to date`: trong 449 sinh viên không học bổng và không đóng phí đúng hạn chỉ **4.5%** tốt nghiệp — phù hợp với "bỏ học → ngừng đóng phí". Vì vậy $T_2$ chỉ là phân tích khám phá.

---

## 4. Học cấu trúc và học tham số

### Structure learning (học cấu trúc / causal discovery)
**Định nghĩa.** Học DAG từ dữ liệu. Ba họ: dựa trên ràng buộc độc lập (PC, FCI), dựa trên hàm điểm (HillClimb, GES), tối ưu liên tục (NOTEARS).

### HillClimbSearch
**Định nghĩa.** Tìm kiếm tham lam trên không gian DAG: mỗi bước thêm/xóa/đảo một cạnh nếu làm tăng hàm điểm, dừng ở cực đại địa phương.
**Trong dự án.** B0 dùng `pgmpy.HillClimbSearch` trên TRAIN.

### Score (hàm điểm): BIC, BDeu, K2
**Công thức.**
- **BIC:** $\text{BIC}(G) = \log \hat L(G; D) - \tfrac{\log N}{2}\, k(G)$, với $k(G) = \sum_j q_j (r_j - 1)$ là số tham số tự do, $N$ số mẫu.
- **BDeu:** log của likelihood biên Bayes với tiên nghiệm Dirichlet đều, tham số $\alpha_{jkl} = ESS / (q_j\, r_j)$:
  $$\log P(D \mid G) = \sum_j \sum_k \Big[\log \tfrac{\Gamma(\alpha_{jk})}{\Gamma(\alpha_{jk} + N_{jk})} + \sum_l \log \tfrac{\Gamma(\alpha_{jkl} + N_{jkl})}{\Gamma(\alpha_{jkl})}\Big]$$
  ($N_{jkl}$: số mẫu có $X_j$ ở trạng thái $l$ và cha ở cấu hình $k$; $\alpha_{jk} = \sum_l \alpha_{jkl}$.)
- **K2:** trường hợp đặc biệt với $\alpha_{jkl} = 1$ cho mọi ô.
**Trong dự án.** Hàm điểm chọn theo log-likelihood VALIDATION; **K2** thắng ở cả 5 seed.

### ExpertKnowledge / forbidden & required edges
**Định nghĩa.** Ràng buộc miền khai báo **trước khi học**: cạnh cấm (blacklist) và cạnh bắt buộc (whitelist).
**Trong dự án.** Cấm $Y \to \{T, Z\}$, cấm $T \to Z$; bắt buộc $T \to Y$ (đây là giả thuyết cần ước lượng, không phải phát hiện).

### Validation log-likelihood
**Công thức.** $\ell_{\text{val}}(G) = \sum_{i \in \text{VAL}} \sum_j \log \hat P\big(x_{ij} \mid \text{pa}(x_{ij})\big)$, với CPD khớp trên TRAIN.
**Trong dự án.** Tiêu chí chọn hàm điểm của B0 và ngưỡng $\tau$, $\lambda_1$ của H2.

### MLE (Maximum Likelihood Estimation) cho CPD
**Công thức.** $\hat P(x_j = l \mid \text{pa} = k) = N_{jkl} / N_{jk}$.
**Hạn chế.** Ô có $N_{jk} = 0$ không xác định; ô có $N_{jkl} = 0$ cho xác suất 0 tuyệt đối ⇒ log-loss vô cực.

### BDeu smoothing (làm mịn Bayes)
**Công thức.**
$$\hat P(x_j = l \mid \text{pa} = k) = \frac{N_{jkl} + ESS/(q_j r_j)}{N_{jk} + ESS/q_j}$$
- $ESS$ (equivalent sample size): độ mạnh tiên nghiệm, như thể có thêm $ESS$ mẫu ảo chia đều.
**Trong dự án.** $ESS = 5$ cho B0 và H2 (cùng họ ước lượng); với $q_Y = 16$, mỗi ô của $Y$ nhận $5/32 \approx 0.16$ mẫu ảo. Cell 11 notebook chính minh họa thêm $ESS = 10$.

### check_model()
**Định nghĩa.** Hàm `pgmpy` kiểm tra mọi CPD khớp với DAG (đúng biến, đúng cha, mỗi cột cộng bằng 1).

---

## 5. Ba mô hình B0, H1, H2

### B0 — Baseline CBN
**Định nghĩa.** DAG học bằng HillClimb trên TRAIN (ràng buộc thời gian, hàm điểm chọn trên VALIDATION) + CPD rời rạc BDeu.
**Trong dự án.** 10–12 cạnh; $\text{Pa}(Y) = \{T, Z_2, Z_3, Z_5\}$; DAG lưu ở `artifacts/graphs/b0_dag_seed<seed>.json`.

### H1 — Neural CPD
**Định nghĩa.** Giữ nguyên DAG B0 của từng run, thay CPD của $Y$ bằng mạng nơ-ron chỉ nhận $\text{Pa}(Y)$; liệt kê mọi cấu hình cha để tạo TabularCPD chuẩn hóa rồi thay vào bản sao B0 (Slide 60).
**Trong dự án.** MLP: Linear(4,32)–BatchNorm–ReLU–Dropout(0.2)–Linear(32,16)–ReLU–Linear(16,1), 769 tham số; checkpoint `artifacts/checkpoints/h1_seed<seed>.pt`.

### MLP (Multi-Layer Perceptron)
**Công thức.** $f(\mathbf{x}) = W_3\, \sigma\big(W_2\, \sigma(W_1 \mathbf{x} + b_1) + b_2\big) + b_3$, với $\sigma$ là hàm kích hoạt (ReLU: $\sigma(u) = \max(0, u)$).

### Binary cross-entropy (BCE)
**Công thức.** $\mathcal{L} = -\frac{1}{n} \sum_i \big[y_i \log p_i + (1 - y_i) \log(1 - p_i)\big]$, $p_i = \text{sigmoid}(f(\mathbf{x}_i))$.

### Early stopping (dừng sớm)
**Định nghĩa.** Dừng huấn luyện khi loss VALIDATION không giảm sau `patience` epoch liên tiếp, giữ trọng số tốt nhất.
**Trong dự án.** patience 15, tối đa 150 epoch.

### Dropout / weight decay
**Định nghĩa.** Dropout tắt ngẫu nhiên tỷ lệ $p$ nơ-ron khi huấn luyện; weight decay cộng $\lambda \lVert \theta \rVert_2^2$ vào loss. Cả hai là regularization chống quá khớp.
**Trong dự án.** Dropout 0.2; weight decay $10^{-4}$.

### H2 — Neural causal discovery (NOTEARS-MLP)
**Định nghĩa.** Học DAG bằng tối ưu liên tục với mạng nơ-ron (Zheng et al., 2020), dùng cùng ràng buộc miền như B0, rồi khớp CPD BDeu trên DAG học được (Slide 62).
**Công thức.** Mỗi nút $j$ có MLP $f_j$; trọng số cạnh $W_{ij} = \lVert \theta^{(1)}_{j, \cdot, i} \rVert_2$ (chuẩn trọng số tầng 1 từ đầu vào $i$ vào MLP của nút $j$).
$$\min_{\theta} \; \tfrac12\, \text{MSE}\big(X, f_\theta(X)\big) + \lambda_1 \lVert \theta^{(1)} \rVert_1 \quad \text{s.t.} \quad h(W) = 0$$
**Trong dự án.** $\lambda_1 = 10^{-4}$ (chọn trên VALIDATION seed 42), 10 nơ-ron ẩn mỗi nút; 7–9 cạnh; mất ≈ 18 s/run trên CPU.

### NOTEARS acyclicity constraint $h(W)$
**Công thức.** $h(W) = \text{tr}\big(e^{W \circ W}\big) - d$
- $W \circ W$: bình phương từng phần tử; $e^{A}$: hàm mũ ma trận; $d$: số nút.
- $h(W) = 0$ **khi và chỉ khi** đồ thị không có chu trình, vì $\text{tr}\big((W \circ W)^k\big)$ cộng dồn trọng số các chu trình độ dài $k$.
**Trong dự án.** Đạt $h \approx 10^{-7}$ ở cả 5 seed (tiêu chí hội tụ $h < 10^{-6}$).

### Augmented Lagrangian
**Công thức.** $L_\rho(\theta, \alpha) = F(\theta) + \alpha\, h(W) + \tfrac{\rho}{2}\, h(W)^2$; vòng trong tối ưu $\theta$, vòng ngoài cập nhật $\alpha \leftarrow \alpha + \rho\, h$, tăng $\rho$ khi $h$ không giảm đủ.
**Trong dự án.** Adam 300 bước/vòng trong; $\rho \times 10$ khi $h$ không giảm 4 lần; tối đa 12 vòng ngoài.

### Domain mask $M$ (mặt nạ cạnh cấm)
**Công thức.** $W_{\text{eff}} = M \odot W$, với $M_{ij} = 0$ nếu cạnh $i \to j$ bị cấm (Slide 63). Quy ước $W[i, j]$ là cạnh $i \to j$.
**Trong dự án.** Dùng đúng danh sách cạnh cấm của B0; code `assert` mọi trọng số bị cấm bằng 0.

### Edge threshold $\tau$ và cycle repair
**Định nghĩa.** Giữ cạnh có $W_{ij} \ge \tau$; nếu còn chu trình, cắt cạnh có trọng số nhỏ nhất trong chu trình (luật khai báo trước — Slide 79).
**Trong dự án.** $\tau \in \{0.1, 0.2\}$ chọn trên VALIDATION; cạnh $T \to Y$ ở H2 **được học**, không bị ép.

### Edge weight ≠ causal effect
**Định nghĩa.** $W_{ij}$ chỉ là đại lượng học nội bộ, không phải hiệu ứng nhân quả hay xác suất cạnh đúng (Slide 58).

---

## 6. Ước lượng hiệu ứng

### ATE (Average Treatment Effect)
**Công thức.** $\text{ATE} = \mathbb{E}[Y \mid do(T = 1)] - \mathbb{E}[Y \mid do(T = 0)] = \mathbb{E}[Y(1) - Y(0)]$
- $Y(t)$: kết cục tiềm năng nếu sinh viên nhận can thiệp $t$.
**Trong dự án (Table B).** $B_0 = +0.2418 \pm 0.0046$, $H_1 = +0.2374 \pm 0.0114$, $H_2 = +0.2358 \pm 0.0067$.

### Standardized risk $q_m(t, C_i)$ và $\text{risk}_m(t)$ (Slide 59, 61)
**Công thức.**
$$q_m(t, C_i) = P_m\big(Y = 1 \mid T = t,\, \text{Pa}(Y)\setminus T = C_i\big), \qquad \text{risk}_m(t) = \frac{1}{n_{\text{test}}} \sum_{i \in \text{TEST}} q_m(t, C_i)$$
$$\widehat{\text{ATE}}_m = \text{risk}_m(1) - \text{risk}_m(0)$$
**Trong dự án.** Bảng $q_m(t, C_i)$ cho vài hồ sơ test hiển thị trực tiếp trong notebook chính (nạp lại DAG và checkpoint).

### G-computation (standardization)
**Định nghĩa.** Khớp mô hình kết cục $\hat\mu(t, \mathbf{z}) = \hat E[Y \mid T = t, \mathbf{Z} = \mathbf{z}]$ rồi lấy trung bình dự đoán khi gán $T = 1$ và $T = 0$ cho mọi người:
$$\widehat{\text{ATE}} = \frac1n \sum_i \big[\hat\mu(1, \mathbf{z}_i) - \hat\mu(0, \mathbf{z}_i)\big]$$

### Linear probability model (LPM)
**Công thức.** $Y = \beta_0 + \beta_T T + \boldsymbol\beta_Z^\top \mathbf{Z} + \varepsilon$; $\hat\beta_T$ là ATE nếu mô hình tuyến tính đúng.

### Outcome regression / T-learner
**Định nghĩa.** Khớp **hai** mô hình riêng $\hat\mu_1$ trên nhóm $T = 1$ và $\hat\mu_0$ trên nhóm $T = 0$; ATE $= \frac1n \sum_i [\hat\mu_1(\mathbf{z}_i) - \hat\mu_0(\mathbf{z}_i)]$.
**Trong dự án.** Bộ ước lượng thứ ba của Table Multi-Treatment trên $C_{\text{test}}$ (trước đây bị gán nhãn sai là AIPW).

### Propensity score
**Công thức.** $e(\mathbf{z}) = P(T = 1 \mid \mathbf{Z} = \mathbf{z})$. Thường được cắt (clip) vào $[0.02, 0.98]$ để tránh trọng số cực đoan.

### IPW (Inverse Probability Weighting)
**Công thức.** $\widehat{\text{ATE}}_{\text{IPW}} = \frac1n \sum_i \Big[\frac{T_i Y_i}{\hat e(\mathbf{z}_i)} - \frac{(1 - T_i) Y_i}{1 - \hat e(\mathbf{z}_i)}\Big]$

### AIPW (Augmented IPW / Doubly Robust)
**Công thức.**
$$\widehat{\text{ATE}}_{\text{AIPW}} = \frac1n \sum_i \Big[\hat\mu_1(\mathbf{z}_i) - \hat\mu_0(\mathbf{z}_i) + \frac{T_i\,(Y_i - \hat\mu_1(\mathbf{z}_i))}{\hat e(\mathbf{z}_i)} - \frac{(1 - T_i)\,(Y_i - \hat\mu_0(\mathbf{z}_i))}{1 - \hat e(\mathbf{z}_i)}\Big]$$
**Tính chất.** Nhất quán nếu **hoặc** mô hình kết cục **hoặc** mô hình propensity đúng. Cần $Y$ của chính mẫu đánh giá.
**Trong dự án.** Chỉ tính trên Full Cohort; trên $C_{\text{test}}$ không dùng $Y_{\text{test}}$ nên dùng T-learner.

### Multi-treatment contrast ($T_1, T_2, T_3$)
**Định nghĩa.** $T_1$ = học bổng; $T_2$ = trạng thái đóng học phí đúng hạn; $T_3$ = $do(T_1 = 1, T_2 = 1)$ so với $do(T_1 = 0, T_2 = 0)$.
**Trong dự án.** Chỉ $T_1$ là kết quả chính. $T_2$ có nguy cơ nhân quả ngược; $T_3 \approx T_1 + T_2$ (không có bằng chứng cộng hưởng) và positivity yếu (37 sinh viên ở ô $T_1 = 1, T_2 = 0$).

---

## 7. Bất định và độ ổn định

### Bootstrap percentile confidence interval
**Định nghĩa.** Lấy mẫu lại có hoàn lại $B$ lần, tính thống kê mỗi lần, lấy phân vị 2.5% và 97.5% làm CI 95%.
**Trong dự án (Table B).** $B = 1000$, lấy mẫu lại các hồ sơ $C_{\text{test}}$, **giữ cố định mô hình** ⇒ chỉ phản ánh bất định lấy mẫu quần thể tham chiếu (rộng ≈ ±0.008), chưa gồm bất định tham số hay bất định đồ thị (Slide 68).

### Refit bootstrap
**Định nghĩa.** Lấy mẫu lại TRAIN và **khớp lại mô hình** mỗi lần, nên phản ánh cả bất định tham số.
**Trong dự án.** Dùng cho G-computation / T-learner của Table Multi-Treatment ($B = 100$).

### Mean ± std qua run
**Định nghĩa.** Trung bình và độ lệch chuẩn của một chỉ số qua 5 seed. Khi split thay đổi, độ tản gồm cả biến thiên của TRAIN và của quần thể tham chiếu (Slide 68).

### Paired difference (hiệu ghép cặp)
**Công thức.** $\Delta_{m - B_0}^{(s)} = \text{Metric}_m^{(s)} - \text{Metric}_{B_0}^{(s)}$ trong cùng seed $s$; báo mean ± std qua seed.

### Paired spread
**Công thức.** $\text{Spread}_m = \max_s \text{ATE}_m^{(s)} - \min_s \text{ATE}_m^{(s)}$.
**Trong dự án (Table C).** B0 0.0087, H1 0.0277, H2 0.0150.

### p-value
**Công thức (kiểm định Wald hai phía).** $p = 2\,\bar\Phi(|z|)$, $z = \widehat{\text{ATE}}/\widehat{\text{SE}}$, $\bar\Phi$ là hàm sống sót của phân phối chuẩn.
**Trong dự án.** Dùng `scipy.stats.norm.sf` thay cho `1 - cdf` để không bị làm tròn về 0.

---

## 8. Đánh giá dự báo

### AUROC
**Định nghĩa.** Diện tích dưới đường ROC (TPR theo FPR khi đổi ngưỡng) = xác suất một mẫu dương ngẫu nhiên được chấm điểm cao hơn một mẫu âm ngẫu nhiên.
**Trong dự án (Table A).** $B_0 = 0.7996$, $H_1 = 0.8012$, $H_2 = 0.8034$.

### Macro-F1
**Công thức.** $F1_c = \dfrac{2\,\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$; Macro-F1 $= \frac12 (F1_{c=0} + F1_{c=1})$. Ngưỡng phân loại chọn trên VALIDATION.

### Precision / Recall
**Công thức.** $\text{Precision} = \dfrac{TP}{TP + FP}$, $\text{Recall} = \dfrac{TP}{TP + FN}$.

### Brier score
**Công thức.** $\text{Brier} = \frac1n \sum_i (p_i - y_i)^2$ — càng nhỏ càng tốt; đo cả độ phân biệt lẫn hiệu chuẩn xác suất.

### Log-loss
**Công thức.** $-\frac1n \sum_i \big[y_i \log p_i + (1 - y_i) \log (1 - p_i)\big]$.

### Calibration (hiệu chuẩn)
**Định nghĩa.** Xác suất dự đoán khớp tần suất thực: trong nhóm được dự đoán 0.7, khoảng 70% thực sự là dương.

### Null finding
**Định nghĩa.** Kết quả không tìm thấy khác biệt đáng kể. Slide 48 yêu cầu báo cáo trung thực các kết quả này.
**Trong dự án.** $\Delta\text{AUROC}_{H_1 - B_0} = +0.0016 \pm 0.0025$, $\Delta_{H_2 - B_0} = +0.0038 \pm 0.0038$ — không vượt độ tản giữa các run ⇒ phần nơ-ron không cải thiện dự báo.

---

## 9. Kiểm định bác bỏ và độ nhạy

### Refuter (kiểm định bác bỏ)
**Định nghĩa.** Thử nghiệm cố tình phá vỡ một giả định để xem ước lượng có phản ứng như kỳ vọng. Vượt qua refuter **không** chứng minh DAG đúng hay không có nhiễu ẩn (Slide 70).

### Placebo treatment refuter
**Định nghĩa.** Hoán vị ngẫu nhiên $T$, ước lượng lại; kỳ vọng hiệu ứng ≈ 0.
**Trong dự án.** DoWhy trên DAG B0 và DAG H2: $+0.0016$; mục 7 notebook chính (Domain DAG): $+0.0047$.

### Data subset refuter
**Định nghĩa.** Ước lượng lại trên 80% dữ liệu lấy ngẫu nhiên nhiều lần; kỳ vọng hiệu ứng gần ước lượng gốc.
**Trong dự án.** B0 $+0.2309 \to +0.2265$; H2 $+0.2262 \to +0.2233$.

### Random common cause refuter
**Định nghĩa.** Thêm một biến nhiễu ngẫu nhiên độc lập làm common cause; kỳ vọng hiệu ứng không đổi.
**Trong dự án.** $+0.2412 \to +0.2411$ (mục 7).

### Custom retraining check (H1)
**Định nghĩa.** Refuter tuyến tính của DoWhy không kiểm tra được mô hình nơ-ron, nên H1 có kiểm định riêng (Slide 70–71): hoán vị $T$ rồi huấn luyện lại MLP (placebo), và huấn luyện lại trên 80% TRAIN (subset).
**Trong dự án.** Placebo $+0.0177$; subset $+0.2365$ (gốc $+0.2453$).

### Partial $R^2$
**Công thức.** $R^2_{Y \sim X \mid W} = \dfrac{\text{RSS}_{\text{không có } X} - \text{RSS}_{\text{có } X}}{\text{RSS}_{\text{không có } X}}$ — tỷ lệ phương sai dư của $Y$ (sau khi đã điều chỉnh $W$) được $X$ giải thích. Với một hệ số hồi quy: $R^2_{Y \sim T \mid \mathbf{Z}} = t^2 / (t^2 + df)$.

### Robustness Value $RV$ (Cinelli & Hazlett, 2020)
**Công thức.**
$$f_q = \frac{q\,|t_T|}{\sqrt{df}}, \qquad RV_q = \frac12\Big(\sqrt{f_q^4 + 4 f_q^2} - f_q^2\Big)$$
- $t_T$: t-stat của hệ số $T$ trong hồi quy $Y \sim T + \mathbf{Z}$; $df$: bậc tự do dư; $q = 1$ nghĩa là đưa ước lượng về 0.
- $RV_{q, \alpha}$: thay $f_q$ bằng $f_q - t^*_{\alpha/2,\, df - 1}/\sqrt{df - 1}$ (mất ý nghĩa thống kê ở mức $\alpha$).
**Diễn giải.** Một biến nhiễu ẩn phải giải thích ít nhất $RV$ phương sai dư của **cả** $T$ và $Y$ mới xóa được hiệu ứng.
**Trong dự án.** $t = 14.20$, $df = 3623$ ⇒ $RV_{q=1} = \mathbf{20.97\%}$, $RV_{q=1, \alpha=0.05} = 18.37\%$. Biến quan sát mạnh nhất ($Z_5$ = học phí đúng hạn trong mã hóa của script độ nhạy) có partial $R^2$ với $Y$ là 10.41%.

---

## 10. Phân tích mở rộng

### CATE (Conditional Average Treatment Effect)
**Công thức.** $\text{CATE}(g) = \mathbb{E}[Y(1) - Y(0) \mid G = g]$ cho nhóm con $g$.
**Trong dự án.** G-computation logistic theo nhóm định trước (không phải Causal Forest): >20 tuổi $+28.02\%$ vs ≤20 tuổi $+16.08\%$; Debtor = 1 $+20.27\%$ vs Debtor = 0 $+21.35\%$. Chưa có CI theo nhóm.

### Relative risk (tỷ số rủi ro)
**Công thức.** $\text{RR} = \text{risk}(1)/\text{risk}(0)$. Khác hiệu ứng tuyệt đối: Debtor = 1 có RR 2.03× nhưng hiệu ứng tuyệt đối không lớn hơn nhóm không nợ.

### Mediation: NDE, NIE (Pearl)
**Công thức.**
$$\text{NDE} = \mathbb{E}\big[Y(1, M(0)) - Y(0, M(0))\big], \qquad \text{NIE} = \mathbb{E}\big[Y(1, M(1)) - Y(1, M(0))\big], \qquad \text{TE} = \text{NDE} + \text{NIE}$$
- $Y(t, m)$: kết cục nếu đặt $T = t$ và $M = m$; $M(t)$: giá trị trung gian dưới $T = t$.
**Trong dự án.** NDE $+16.95\%$ (78.5%), NIE $+4.65\%$ (21.5%) qua tín chỉ HK1.

### Policy value và phân bổ theo CATE
**Công thức.** $V(\pi) = \frac1n \sum_i \big[\pi(\mathbf{z}_i)\, \hat\mu_1(\mathbf{z}_i) + (1 - \pi(\mathbf{z}_i))\, \hat\mu_0(\mathbf{z}_i)\big]$, $\pi(\mathbf{z}) \in \{0, 1\}$ với ràng buộc ngân sách $\sum_i \pi(\mathbf{z}_i) = 969$.
**Trong dự án.** +164 sinh viên tốt nghiệp **dự kiến** — ước lượng in-sample, chưa cross-fit ⇒ ủng hộ có điều kiện.

### Out-of-distribution (OOD) transfer
**Định nghĩa.** Áp mô hình học trên một phân phối sang phân phối khác.
**Trong dự án.** Kaggle S4E6 là dữ liệu **tổng hợp** sinh từ UCI; kết quả trái chiều (UCI-trained 0.215 vs Kaggle-trained 0.30–0.32) ⇒ không chứng minh bất biến.

### Ablation (A1–A3)
**Định nghĩa.** Bỏ đúng **một** thành phần để đo đóng góp của nó: A1 bỏ dropout của H1, A2 bỏ một ràng buộc của H2, A3 dùng MLE thay BDeu (Slide 64–67, tùy chọn).
**Trong dự án.** Chạy bằng `tools/run_ablations_and_extensions.py` với Domain DAG cố định — chỉ đọc như phụ lục.

---

## 11. Thư viện và artifact

| Từ khóa | Định nghĩa | Trong dự án |
| :--- | :--- | :--- |
| **pgmpy** | Thư viện Python cho mạng Bayes: học cấu trúc, CPD, suy diễn | 1.1.2 — HillClimbSearch, BDeu, `DiscreteBayesianNetwork` |
| **DoWhy** | Thư viện suy luận nhân quả 4 bước Model → Identify → Estimate → Refute | 0.14 — identify trên từng DAG, refuter |
| **PyTorch** | Thư viện học sâu | 2.15 (CPU) — MLP của H1, NOTEARS-MLP của H2 |
| **Table A** | So sánh dự báo trên TEST (Slide 74) | `artifacts/tables/Table_A_Prediction.csv` |
| **Table B** | Can thiệp chuẩn hóa trên $C_{\text{test}}$: risk, ATE, CI, tập điều chỉnh, overlap (Slide 75) | `artifacts/tables/Table_B_Intervention.csv` |
| **Table C** | Refuter, độ tản, runtime, phần cứng (Slide 75) | `artifacts/tables/Table_C_Robustness_Cost.csv` |
| **Checkpoint** | File trọng số + cấu hình mô hình đã huấn luyện | `artifacts/checkpoints/h1_seed<seed>.pt` |
| **Graph artifact** | Cạnh DAG, lựa chọn trên VALIDATION, kết quả identify | `artifacts/graphs/b0_dag_seed<seed>.json`, `h2_dag_seed<seed>.json`, `h2_weights_seed<seed>.npy` |
| **Vectorized CPD query** | Đọc $q_m(t, C_i)$ bằng numpy advanced indexing trực tiếp trên mảng CPD | ≈ 0.2 ms cho 545 hồ sơ |
