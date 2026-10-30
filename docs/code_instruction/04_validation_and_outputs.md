# Kiểm thử, output và tiêu chí hoàn thành

## Unit checks tối thiểu

- Synthetic generator tạo đúng cột và miền giá trị.
- B0 `check_model()` pass.
- `p_do(1) - p_do(0)` trên synthetic gần `0.30` trong sai số lấy mẫu.
- `P(Y|T)` và `P(Y|do(T))` được tính bởi hai đường code khác nhau.
- UCI cohort có 3,630 dòng sau lọc.
- không có `Enrolled` trong outcome.
- split không giao nhau và phủ đủ cohort.
- H2 mask không cho phép cạnh bị cấm.
- A1/A2/A3 chỉ khác đúng một cấu hình.
- refutation output được lưu, không bị nuốt exception.

## Output directory đề xuất

```text
artifacts/
  data_audit.json
  splits/seed_42.json
  splits/seed_123.json
  checkpoints/
  graphs/
    domain_dag.png
    h2_adjacency_seed_42.png
  results/
    synthetic_results.csv
    uci_results.csv
    ablation_results.csv
    refutations.json
  tables/
    table_a.csv
    table_b.csv
    table_c.csv
```

## Bảng A/B/C phải có

**Bảng A - UCI:** model, identification status, adjustment set, p_do(0),
p_do(1), ATE và CI, AUROC, Macro-F1, Brier, runtime.

**Bảng B - Synthetic:** N, estimated ATE, true ATE `0.300`, absolute error,
std qua seeds, graph error nếu có, runtime.

**Bảng C - Ablation:** parent, variant, changed component, paired parent
effect, paired variant effect, delta và p-value/CI.

## Điều kiện chấp nhận trước khi nộp

- `Run all` trên kernel mới không lỗi.
- Tất cả kết quả đều truy được về seed và split.
- Không có kết quả placeholder `...`, số 0 giả hoặc success-shaped fallback.
- Có log trường hợp H2/A3 fail nếu xảy ra.
- Notebook hiển thị rõ observation vs intervention.
- Có requirements/environment file và artifacts máy đọc được.
- Không kết luận “neural luôn tốt hơn”; kết luận phải dựa trên mean, spread,
  uncertainty và refutation.

