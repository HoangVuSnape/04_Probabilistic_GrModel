# ĐỊNH HƯỚNG NGHIÊN CỨU BÁM REQUIREMENTS MIDTERM & FINAL
## Causal Bayesian Network + Deep Learning cho can thiệp học bổng

> **Ngày rà soát:** 2026-10-29  
> **Tài liệu tham chiếu:** `docs/requirements/midterm/`, `docs/requirements/final/`, `research/01`–`research/08`, `report/decisions/ADR_001_dataset_selection.md`.  
> **Mục đích:** Chuyển rubric của môn học thành một thiết kế nghiên cứu và thực nghiệm có thể triển khai, demo và tái lập.

---

## 1. Hiểu đúng mục tiêu của môn học

Đề tài không phải là một cuộc thi phân loại thông thường. Accuracy chỉ là một phần của đánh giá. Câu hỏi trung tâm phải là:

> Nếu nhà trường can thiệp cấp học bổng/hỗ trợ tài chính (`do(T=1)`) thay vì không cấp (`do(T=0)`), xác suất sinh viên tốt nghiệp thay đổi bao nhiêu sau khi điều chỉnh các yếu tố nền?

Vì vậy, toàn bộ quy trình phải tách thành ba giai đoạn:

1. **Discovery:** lựa chọn biến, trật tự thời gian và đề xuất Domain DAG.
2. **Identification:** chứng minh tập điều chỉnh thỏa tiêu chuẩn Backdoor.
3. **Estimation:** ước lượng $P(Y=1\mid do(T=t))$, ATE, rủi ro và độ bất định.

Mạng nơ-ron không thay thế bước định danh. Một mô hình học tốt $P(Y\mid X)$ chưa chứng minh được chiều nhân quả hoặc hiệu ứng can thiệp.

---

## 2. Vai trò của từng bộ dữ liệu

| Bộ dữ liệu | Vai trò trong nghiên cứu | Không được diễn giải thành |
|---|---|---|
| UCI 697 | Dữ liệu thực nghiệm chính cho câu hỏi causal và Bảng A | Bằng chứng nhân quả tự động |
| Synthetic Tutoring Benchmark | Bài kiểm tra có chân lý biết trước, bắt buộc cho Bảng B | Đại diện cho sinh viên thật |
| Kaggle S4E6 | Benchmark synthetic/robustness phụ, kiểm tra distribution shift | 76.000 sinh viên độc lập |

Kaggle S4E6 chỉ được đưa vào sau khi pipeline UCI và synthetic truth chạy đúng. Nếu dùng Kaggle để huấn luyện hoặc bổ sung, phải ghi rõ đó là `synthetic augmentation`, không gộp kết quả với UCI mà không có nhãn nguồn.

Chi tiết bản chất Kaggle được lưu tại [08_kaggle_s4e6_data_deep_dive.md](./08_kaggle_s4e6_data_deep_dive.md).

---

## 3. Câu hỏi, biến và thời gian

### 3.1. Biến causal chính

- $T$: `Scholarship holder`, treatment nhị phân.
- $Y$: kết quả nhị phân sau khi loại `Enrolled`: `Graduate=1`, `Dropout=0`.
- $\mathbf{Z}$: tập biến nền được chọn trước treatment.

Tập $\mathbf{Z}$ ban đầu:

$$
\mathbf{Z}=
\{\text{Admission grade},\text{Debtor},\text{Tuition fees up to date},
\text{Mother's qualification},\text{Age at enrollment}\}.
$$

### 3.2. Hai prediction estimand phải tách riêng

**Admission-time setting:** chỉ dùng biến có thể biết trước hoặc tại thời điểm nhập học. Đây là setting phù hợp với mục tiêu cảnh báo sớm.

**Semester-aware setting:** cho phép biến học kỳ 1 hoặc 2. Đây là dự báo cập nhật trong quá trình học, không được gọi là dự báo tại thời điểm nhập học.

Các biến `Curricular units ... grade/approved` phải được gắn nhãn hậu kỳ. Nếu dùng chúng trong outcome model causal, cần phân tích chúng như mediator hoặc biến hậu treatment, không đưa tùy tiện vào tập Backdoor adjustment.

---

## 4. Domain DAG và nhận diện

### 4.1. DAG B0

Các cạnh cốt lõi:

$$
Z_i\rightarrow T,\quad Z_i\rightarrow Y,\quad T\rightarrow Y.
$$

Không cho phép:

- $Y\rightarrow T$;
- $Y\rightarrow Z_i$;
- $T\rightarrow Z_i$;
- mọi cạnh đi ngược trật tự thời gian.

Factorization:

$$
P(\mathbf Z,T,Y)
=\left(\prod_iP(Z_i)\right)
P(T\mid\mathbf Z)P(Y\mid T,\mathbf Z).
$$

Sau can thiệp:

$$
P(Y\mid do(T=t))
=\sum_{\mathbf z}P(Y\mid T=t,\mathbf z)P(\mathbf z).
$$

ATE:

$$
\mathrm{ATE}
=P(Y=1\mid do(T=1))
-P(Y=1\mid do(T=0)).
$$

### 4.2. Ba giả định bắt buộc phải bảo vệ

1. **No unmeasured confounding / ignorability:** $Y(t)\perp T\mid\mathbf Z$.
2. **Consistency:** outcome quan sát được dưới treatment thực tế tương ứng với potential outcome của treatment đó.
3. **Positivity:** $0<P(T=t\mid\mathbf Z=\mathbf z)<1$ trên miền cần suy luận.

Đây là giả định nghiên cứu, không phải sự thật được chứng minh chỉ bằng accuracy. Phải báo cáo giới hạn của nhiễu ẩn và khả năng overlap yếu.

---

## 5. Thiết kế ba mô hình bắt buộc

| Mô hình | Cố định | Thành phần thay đổi | Câu hỏi kiểm chứng |
|---|---|---|---|
| B0 | Domain DAG | Bayesian CPD, BDeu/ESS=5 | Mốc chuẩn có smoothing |
| H1 | Domain DAG | CPD của $Y$ bằng MLP PyTorch | Neural mechanism có cải thiện estimation không? |
| H2 | Domain mask + acyclicity | Học $W$ bằng gradient, CPD non-neural | Học cấu trúc có cải thiện mà vẫn hợp lý causal không? |
| H1+H2 | Cấu trúc H2 | CPD neural của H1 | Có lợi ích kết hợp hay không? |

### 5.1. B0

- `pgmpy.DiscreteBayesianNetwork`.
- `BayesianEstimator` với BDeu prior, ESS=5.
- Tính CPD cho mọi node.
- Truy vấn intervention bằng truncated factorization hoặc do-operation.
- Không sử dụng MLE làm mô hình chính vì sparse cells có thể tạo xác suất 0.

### 5.2. H1

- Giữ nguyên Domain DAG.
- MLP chỉ nhận các parent của $Y$.
- BCE loss, Adam, dropout và weight decay.
- Early stopping chỉ dùng validation.
- Chuyển xác suất MLP thành CPD hợp lệ trước khi suy luận intervention.

H1 chỉ thay cơ chế tham số; không được đồng thời thay DAG, split hoặc preprocessing. Như vậy chênh lệch B0/H1 mới có ý nghĩa.

### 5.3. H2

Tối ưu ma trận liên tục $W$:

$$
\min_W \mathcal L(W;X)+\lambda\lVert W\rVert_1
\quad\text{s.t.}\quad
h(W)=\operatorname{tr}\left(e^{W\odot W}\right)-d=0.
$$

Áp dụng domain mask:

$$
W_{\mathrm{eff}}=M\odot W.
$$

Quy trình phải lưu lại:

1. loss theo epoch;
2. giá trị acyclicity $h(W)$;
3. ma trận $W_{\mathrm{eff}}$;
4. ngưỡng cắt cạnh $\tau$;
5. DAG trước và sau threshold;
6. kiểm tra cycle;
7. cạnh học được có vi phạm temporal order hay không.

H2 thất bại nếu không hội tụ, sinh cycle sau threshold hoặc không thể nhận diện tập adjustment. Khi đó phải báo cáo `Failed/Not identified`, không thay bằng một kết quả thành công-shaped.

---

## 6. Ablation đúng nguyên tắc “one change only”

| Ablation | Parent | Chỉ thay đổi | Mục tiêu |
|---|---|---|---|
| A1 | H1 | Dropout=0 và bỏ weight decay theo đúng cấu hình đã định | Đo vai trò regularization |
| A2 | H2 | Mở đúng một cạnh bị cấm, ví dụ $M_{Y,Z_1}=1$ | Đo giá trị domain mask |
| A3 | B0 | BayesianEstimator → MaximumLikelihoodEstimator | Đo độ nhạy sparse cells/pseudo-count |

Mỗi ablation phải dùng cùng seed, split, preprocessing, epoch budget và metric với parent. Báo cáo:

$$
\Delta=\text{metric}_{variant}-\text{metric}_{parent}.
$$

Không được thay nhiều thành phần cùng lúc rồi gọi đó là ablation.

---

## 7. Giao thức đối sánh công bằng

### 7.1. Split và seeds

- Train/Validation/Test: 70/15/15.
- Seeds: `[42, 123, 456, 789, 999]`.
- Stratify theo outcome; nếu khả thi, kiểm tra thêm phân bố treatment.
- Fit mọi discretizer/encoder chỉ trên Train.
- Test chỉ mở ở bước đánh giá cuối.

### 7.2. Metrics bắt buộc

Prediction:

- AUROC.
- Macro-F1.
- Brier score.
- Accuracy chỉ là metric phụ theo Kaggle.

Causal:

- $P(Y=1\mid do(T=0))$.
- $P(Y=1\mid do(T=1))$.
- ATE/risk difference.
- Mean, standard deviation và 95% CI qua 5 seeds.
- Runtime.

Không báo cáo một con số đơn lẻ nếu đó là kết quả cuối cùng của nghiên cứu.

---

## 8. Synthetic truth và DoWhy

### 8.1. Synthetic Tutoring Benchmark

Phải tạo một SCM có chân lý:

$$
P(Y=1\mid do(T=0))=0.30,\qquad
P(Y=1\mid do(T=1))=0.60,
$$

do đó:

$$
\mathrm{ATE}_{true}=0.30.
$$

Benchmark dùng để phát hiện lỗi code trong intervention và estimation. Nếu mô hình không thu hồi được ATE gần 0.30, chưa được dùng kết quả UCI để kết luận.

### 8.2. Hai refutation bắt buộc

1. **Placebo treatment refuter**
   - Xáo treatment hoặc thay treatment giả.
   - Kỳ vọng hiệu ứng gần 0.
   - Báo cáo estimate, phân phối refutation và p-value nếu thư viện cung cấp.

2. **Data subset refuter**
   - Lặp trên subset, ví dụ 80%.
   - Kỳ vọng estimate ổn định trong khoảng bất định.

Ngoài con số pass/fail, phải giải thích refuter đang kiểm tra giả định nào và không chứng minh được điều gì. Refutation không thể chứng minh không có hidden confounding.

### 8.3. DoWhy GCM

Dùng GCM để mô phỏng can thiệp và đối soát với kết quả CPD/BN. Lưu seed, sample size, model graph và kết quả máy đọc được.

---

## 9. Cấu trúc kết quả phải có

### Bảng A - UCI

Mỗi hàng B0/H1/H2/(H1+H2), mỗi metric là mean ± std qua 5 seeds:

- identification status;
- adjustment set;
- $P(Y=1\mid do(T=0))$;
- $P(Y=1\mid do(T=1))$;
- ATE và 95% CI;
- AUROC, Macro-F1, Brier;
- runtime.

### Bảng B - Synthetic truth

- sample size;
- estimated ATE;
- true ATE=0.30;
- absolute error;
- standard deviation;
- graph edit distance nếu có H2;
- runtime.

### Bảng C - Ablations

- parent;
- variant;
- metric parent;
- metric variant;
- $\Delta$;
- paired seed comparison;
- kết luận ngắn.

Kaggle benchmark nên là bảng/phụ lục riêng, không thay thế Bảng A UCI hoặc Bảng B synthetic truth.

---

## 10. Kịch bản demo Midterm 6 bước

Notebook cần được sắp xếp theo đúng trình tự:

1. Câu hỏi causal, $T/Y/Z$, temporal order, DAG B0 và ba giả định.
2. CPD B0; so sánh $P(Y\mid T)$ với $P(Y\mid do(T))$.
3. Mở code H1 và H2; chỉ ra MLP, mask, $W$, acyclicity.
4. Đổi trực tiếp $T=0\rightarrow T=1$ và chạy lại intervention risk.
5. Hiển thị Bảng A, Bảng B, Bảng C và DoWhy refutations.
6. Trình bày một failure case và truy ngược tới cell/code tạo ra kết quả.

Notebook phải có saved outputs, Table of Contents và không có cell lỗi. Mọi thành viên phải nộp cùng notebook theo yêu cầu Midterm.

---

## 11. Đóng gói Final và tái lập

Final yêu cầu:

- PDF biên dịch từ LaTeX.
- `.tex`, `.bib`, figures và tables.
- Python/Notebook chạy từ đầu trong môi trường sạch.
- `requirements.txt` hoặc `environment.yml`.
- artifacts `.json`/`.csv`.
- AI-use statement.
- Trích dẫn UCI, Pearl/do-calculus, NOTEARS hoặc DAG-GNN, DoWhy và pgmpy.

Đề xuất cấu trúc artifacts:

```text
artifacts/
├── data_audit.json
├── overlap_by_seed.csv
├── model_results_by_seed.csv
├── causal_effects_by_seed.csv
├── synthetic_truth_results.csv
├── ablation_results.csv
├── refutation_results.json
└── learned_graphs/
    ├── h2_seed42.json
    └── ...
```

Mọi số trong LaTeX phải truy nguyên được về artifacts hoặc log. Không nhập tay kết quả sau khi xem notebook.

---

## 12. Các điểm cần sửa/kiểm soát trước khi triển khai

1. **Không dùng Kaggle synthetic làm causal dataset chính.** Điều này phù hợp với bản chất dữ liệu và tránh phóng đại cỡ mẫu.
2. **Xác định chính xác thời điểm `Scholarship holder`.** Nếu thời điểm thực tế không đủ rõ, phải ghi là treatment quan sát được tại thời điểm đầu kỳ và thảo luận measurement timing.
3. **Kiểm tra lại tập $\mathbf Z$.** Chỉ đưa biến vào Backdoor set khi có lý do temporal/domain; không mặc định mọi feature là confounder.
4. **Không tự động coi `Tuition fees up to date` là baseline.** Nếu biến được ghi nhận sau khi cấp học bổng, nó có thể là mediator hoặc post-treatment variable.
5. **H2 không được chỉ vẽ DAG theo gợi ý.** Phải có optimizer, gradient, mask, acyclicity và log.
6. **Nếu H2 không nhận diện được effect, báo cáo thất bại.** Không ép threshold hoặc sửa graph thủ công để làm đẹp kết quả.
7. **DoWhy refutation là kiểm tra độ bền, không phải chứng minh causal truth.**
8. **Bảng Kaggle phải tách khỏi bảng UCI.** Accuracy Kaggle không thay thế ATE trên UCI.

---

## 13. Ưu tiên triển khai tiếp theo

### P0 - Bắt buộc trước khi huấn luyện sâu

- Chốt data dictionary và temporal status cho từng feature.
- Xây data audit: shape, missingness, class balance, treatment balance, overlap.
- Tạo Synthetic Tutoring Benchmark và xác nhận true ATE=0.30.
- Viết hàm intervention chung dùng cho B0/H1/H2.

### P1 - Đủ điều kiện Midterm

- B0 chạy 5 seeds.
- H1 chạy 5 seeds.
- H2 chạy trên synthetic truth trước, sau đó trên UCI.
- A1/A2/A3 dùng paired seeds.
- Hai DoWhy refuters.
- Xuất ba bảng và runtime.

### P2 - Robustness và Final

- Kaggle distribution audit.
- UCI↔Kaggle transfer.
- Exact/near-duplicate audit.
- LaTeX và artifacts tự động.
- Failure analysis và sensitivity analysis cho hidden confounding/overlap.

---

## 14. Kết luận định hướng

Đường hướng phù hợp với requirements là:

> **UCI causal question → Domain DAG và Backdoor identification → Synthetic truth validation → B0/H1/H2 controlled comparison → DoWhy refutation → Kaggle synthetic robustness benchmark → báo cáo hạn chế và tái lập.**

Không nên đảo thành:

> Kaggle accuracy → suy ra hiệu quả học bổng.
