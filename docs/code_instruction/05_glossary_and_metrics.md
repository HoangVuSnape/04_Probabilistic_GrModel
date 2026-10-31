# Glossary và metric guide

Tài liệu này giải thích các keyword xuất hiện trong code, notebook và báo cáo.
Ví dụ dùng `T=Scholarship holder`, `Y=Graduate`.

## 1. Causal keywords

| Keyword | Nghĩa và dùng để làm gì | Ví dụ ngắn |
|---|---|---|
| **Causal question** | Câu hỏi về tác động của một hành động, không chỉ tương quan. | “Nếu cấp học bổng cho sinh viên thì tỷ lệ tốt nghiệp thay đổi bao nhiêu?” |
| **Treatment (`T`)** | Biến được can thiệp. | `T=1`: có học bổng; `T=0`: không có. |
| **Outcome (`Y`)** | Kết quả cần đo. | `Y=1`: Graduate; `Y=0`: Dropout. |
| **Confounder (`Z`)** | Biến xảy ra trước `T`, ảnh hưởng cả `T` và `Y`, gây nhiễu khi so sánh. | `Debtor` có thể ảnh hưởng khả năng nhận hỗ trợ và nguy cơ bỏ học. |
| **DAG** | Đồ thị có hướng không chu trình, biểu diễn giả định nguyên nhân. | `Debtor -> T -> Y` và `Debtor -> Y`. |
| **Domain DAG** | DAG được đề xuất từ thời gian và kiến thức miền. | Biết trước học bổng không thể bị gây ra bởi kết quả tốt nghiệp tương lai. |
| **Edge** | Một cạnh `A -> B` nghĩa là A được giả định là nguyên nhân của B. | `Scholarship holder -> Target`. |
| **Acyclic** | Không thể đi theo các cạnh và quay lại node ban đầu. | Có `T -> Y` thì không thêm `Y -> T`. |
| **Observation** | Quan sát nhóm thực tế, chưa ép treatment. | `P(Y=1 \| T=1)`: tỷ lệ tốt nghiệp của nhóm có học bổng. |
| **Intervention / do-operator** | Ép treatment bằng chính sách hoặc hành động. | `P(Y=1 \| do(T=1))`: cấp học bổng cho mọi sinh viên trong mô phỏng. |
| **Backdoor criterion** | Điều kiện chọn tập `Z` để chặn đường gây nhiễu đi vào treatment. | Điều chỉnh theo các biến baseline trước `T`. |
| **Adjustment set** | Tập biến dùng trong backdoor adjustment. | `Z={Admission grade, Debtor, ...}`. |
| **Identification** | Kiểm tra hiệu ứng causal có suy ra được từ dữ liệu quan sát theo DAG hay không. | DoWhy trả về identified bằng backdoor. |
| **Estimation** | Tính giá trị hiệu ứng sau khi đã định danh. | Ước lượng `p_do(0)`, `p_do(1)` và ATE. |
| **Positivity / overlap** | Mỗi nhóm baseline cần có cơ hội nhận cả `T=0` và `T=1`. | Trong cùng nhóm `Z`, phải có sinh viên có và không có học bổng. |
| **Consistency** | Treatment quan sát thực tế tương ứng với potential outcome của treatment đó. | Sinh viên thực sự nhận học bổng có outcome tương ứng với `Y(1)`. |
| **Unmeasured confounding** | Nhiễu quan trọng không có trong dữ liệu. | Động lực học tập không đo được có thể làm sai ATE. |
| **Leakage** | Dùng thông tin xuất hiện sau thời điểm dự báo/can thiệp. | Dùng điểm học kỳ 2 để điều chỉnh cho học bổng đầu kỳ. |

## 2. Mô hình và thực nghiệm

| Keyword | Nghĩa và dùng để làm gì | Ví dụ ngắn |
|---|---|---|
| **B0** | Baseline: DAG cố định, CPD Bayesian smoothing. | B0 dùng `BayesianEstimator(BDeu, ESS=5)`. |
| **H1** | Giữ DAG B0, thay CPD của outcome bằng neural network. | MLP học `P(Y \| T,Z)`. |
| **H2** | Học cấu trúc DAG bằng gradient, acyclicity penalty và domain mask. | H2 học ma trận cạnh `W`. |
| **CPD** | Conditional Probability Distribution, xác suất của node khi biết parents. | `P(Y \| T,Z)`. |
| **BDeu / ESS** | Bayesian prior làm trơn CPD; ESS là equivalent sample size. | ESS=5 giúp giảm xác suất 0 do ô thưa. |
| **Ablation** | Bỏ đúng một thành phần để đo vai trò của thành phần đó. | A1 bỏ dropout khỏi H1. |
| **Seed** | Số khởi tạo ngẫu nhiên để kiểm tra độ ổn định. | Chạy `[42,123,456,789,999]`. |
| **Train / validation / test** | Train để fit, validation để chọn cấu hình, test để đánh giá cuối. | 70/15/15, test không được dùng để chọn model. |
| **Synthetic benchmark** | Dữ liệu mô phỏng có causal truth biết trước. | True ATE = 0.30. |
| **Refutation** | Kiểm tra độ bền của kết quả causal trước một thay đổi giả định. | Placebo treatment nên cho effect gần 0. |
| **GCM** | Graphical Causal Model trong DoWhy để mô phỏng can thiệp. | Ép `T=0` và `T=1`, so sánh tỷ lệ `Y=1`. |
| **Acyclicity penalty `h(W)`** | Phạt chu trình trong H2. | Sau tối ưu cần `h(W)` gần 0. |
| **Domain mask** | Cấm cạnh trái với temporal/domain knowledge. | Cấm `Y -> T` và `T -> Z`. |

## 3. Causal metrics

| Metric / keyword | Dùng gì và ý nghĩa | Ví dụ ngắn |
|---|---|---|
| **`p_do_t0`** | `P(Y=1 \| do(T=0))`, rủi ro outcome khi không can thiệp. | `0.42` nghĩa là dự kiến 42% tốt nghiệp nếu không cấp học bổng. |
| **`p_do_t1`** | `P(Y=1 \| do(T=1))`, rủi ro outcome khi can thiệp. | `0.55` nghĩa là dự kiến 55% tốt nghiệp nếu cấp học bổng. |
| **ATE** | Average Treatment Effect: `p_do_t1 - p_do_t0`. Đo thay đổi trung bình do can thiệp. | `0.55-0.42=0.13`: tăng 13 điểm phần trăm. |
| **Risk difference** | Tên khác của ATE khi outcome là xác suất/rủi ro. | `0.13` là chênh lệch rủi ro tuyệt đối. |
| **Absolute error** | `abs(estimated ATE - true ATE)`, dùng cho synthetic. | Estimate `0.27`, truth `0.30` -> error `0.03`. |
| **Confidence interval (CI)** | Khoảng bất định của ước lượng, thường 95%. | ATE `0.13`, 95% CI `[0.04,0.22]`: khoảng vẫn dương. |
| **Standard deviation (std)** | Độ phân tán kết quả qua nhiều seed. | Mean ATE `0.13`, std `0.02` ổn định hơn std `0.15`. |
| **Refutation effect** | Hiệu ứng sau kiểm định bác bỏ. | Placebo effect gần `0` là dấu hiệu tốt, không phải proof tuyệt đối. |
| **Graph Edit Distance (GED)** | Số phép thêm/xóa/đổi cạnh để biến DAG học được thành DAG thật. | GED `1` tốt hơn GED `5`. |
| **Ablation delta (`Δ`)** | `metric_variant - metric_parent`, đo tác động của thay đổi. | A1 ATE `0.10`, H1 `0.13` -> `Δ=-0.03`. |

## 4. Prediction metrics

Các metric này đánh giá dự báo ngoài mẫu; chúng **không tự chứng minh quan hệ
nhân quả**.

| Metric | Dùng gì và ý nghĩa | Ví dụ ngắn |
|---|---|---|
| **AUROC** | Khả năng xếp hạng positive cao hơn negative trên mọi threshold. | `0.80` tốt hơn `0.60`; `0.50` gần ngẫu nhiên. |
| **Macro-F1** | Trung bình F1 của từng class, coi hai class có trọng số ngang nhau. | Hữu ích khi Dropout/Graduate lệch tỷ lệ. |
| **Precision** | Trong các mẫu dự đoán Graduate, bao nhiêu mẫu thật là Graduate. | 80/100 đúng -> precision `0.80`. |
| **Recall** | Trong các Graduate thật, mô hình tìm được bao nhiêu. | Tìm 90/120 -> recall `0.75`. |
| **F1** | Trung bình điều hòa precision và recall. | Precision=.8, recall=.5 -> F1≈`.62`. |
| **Brier score** | Mean squared error giữa xác suất dự báo và nhãn thật; càng thấp càng tốt. | Dự báo `.9` cho nhãn `0` bị phạt mạnh hơn dự báo `.6`. |
| **Runtime** | Thời gian chạy model/pipeline, dùng đánh giá chi phí tính toán. | H2 mất 120 giây, B0 mất 2 giây. |

## 5. Cách đọc kết quả mẫu

```text
Model: B0
p_do_t0 = 0.42
p_do_t1 = 0.55
ATE = 0.13
95% CI = [0.04, 0.22]
AUROC = 0.78
Macro-F1 = 0.70
Brier = 0.18
```

Diễn giải: theo DAG và các giả định đã nêu, can thiệp học bổng làm tăng xác
suất tốt nghiệp ước tính 13 điểm phần trăm. CI không chứa 0 trong ví dụ này,
nhưng AUROC/Macro-F1 chỉ nói về chất lượng dự báo; cần xem thêm overlap,
refutation và hidden confounding trước khi đưa ra khuyến nghị chính sách.

