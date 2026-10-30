# Các mô hình phải cài đặt

## B0 - Baseline CBN

1. Tạo `DiscreteBayesianNetwork` với các cạnh:
   `Zi -> T`, `Zi -> Y`, `T -> Y`.
2. Fit `BayesianEstimator`, `prior_type="BDeu"`,
   `equivalent_sample_size=5`.
3. Gọi `check_model()`, nếu fail thì raise lỗi có ngữ cảnh.
4. Dùng `VariableElimination` để tính `P(Y|T)` và công thức adjustment
   `P(Y|do(T=t))`.
5. Xuất CPD, DAG, p_do(0), p_do(1), ATE.

Synthetic benchmark phải kiểm tra gần các giá trị lý thuyết:

```text
P(Y=1 | do(T=0)) = 0.30
P(Y=1 | do(T=1)) = 0.60
ATE = 0.30
```

## H1 - Neural parameters

- Giữ nguyên DAG B0.
- Mạng MLP chỉ nhận `T` và các parents của `Y` (không nhận hậu duệ).
- Output là xác suất `P(Y=1 | T, Z)`.
- Dùng BCE, validation early stopping và lưu checkpoint tốt nhất.
- H1 phải dùng cùng split với B0.
- Sau train, dùng output xác suất trong cùng công thức adjustment; không
  thay do-query bằng `P(Y|T)`.

## H2 - Neural structure

- Dùng NOTEARS/DAG-GNN hoặc triển khai tương đương có tối ưu gradient liên
  tục.
- Có ma trận trọng số cạnh `W`, acyclicity penalty `h(W)` và domain mask.
- Mask phải cấm các cạnh vi phạm thời gian: `Y -> T`, `Y -> Zi`, `T -> Zi`.
- Báo cáo `h(W)`, số cạnh sau threshold, DAG và graph edit distance nếu có
  ground truth.
- Nếu không hội tụ hoặc sinh chu trình, đánh dấu `Failed`; không tự động coi
  là DAG hợp lệ.

## Ablations - chỉ thay một yếu tố

| Mã | Parent | Thay đổi duy nhất |
|---|---|---|
| A1 | H1 | dropout/regularization = 0 |
| A2 | H2 | mở đúng một cạnh bị cấm trong temporal mask |
| A3 | B0 | thay BDeu bằng `MaximumLikelihoodEstimator` |

Mỗi ablation phải dùng cùng seed, split, epochs/budget và metric với parent.
Xuất `delta = variant - parent` theo từng seed rồi mới tính mean/std/CI.

## DoWhy và GCM

Cho UCI và synthetic:

1. Tạo `CausalModel` với graph rõ ràng.
2. `identify_effect(proceed_when_unidentifiable=False)`.
3. Estimate backdoor ATE và confidence interval.
4. Chạy:
   - `placebo_treatment_refuter`;
   - `data_subset_refuter`.
5. Dùng DoWhy GCM để đặt `T` hằng lần lượt bằng 0 và 1, sau đó so sánh
   rủi ro outcome.

