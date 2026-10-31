# Kiểm thử, output và tiêu chí hoàn thành
> Cập nhật theo Slide 51, 53, 74–75, 76, 79 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

Giải thích keyword, công dụng metric và ví dụ diễn giải xem tại
[05_glossary_and_metrics.md](./05_glossary_and_metrics.md).

## Unit checks tối thiểu

- UCI data audit đầy đủ: URL, checksum, lưu đồ lọc (3,630 dòng sau loại `Enrolled`).
- Không có rò rỉ nhãn kết quả vào tập dự báo.
- Trật tự thời gian được tôn trọng: các biến nền $C$ đo trước $T$, $T$ đo trước $Y$.
- B0 `check_model()` pass.
- H1 MLP chỉ nhận đầu vào là parents của $Y$; bảng CPD chuẩn hóa được kiểm tra tổng bằng 1.
- H2 ma trận mặt nạ $M$ đảm bảo triệt tiêu các cạnh cấm thời gian; ràng buộc chu trình được thỏa mãn.
- `P(Y|T)` (quan sát) và `P(Y|do(T))` (can thiệp) được tính bởi hai đường code khác nhau.
- Chuẩn hóa can thiệp thực hiện trên cùng tập hiệp biến nền kiểm thử $C_{\text{test}}$, không dùng nhãn thực tế $Y_{\text{test}}$ để tính hiệu ứng.
- Ít nhất 5 lần chạy ghép cặp (paired runs $\ge 5$) dùng chung phân chia dữ liệu.
- Ít nhất 2 bài kiểm tra bác bỏ DoWhy: `placebo_treatment_refuter` và `data_subset_refuter`.
- Báo cáo và ghi log các trường hợp thất bại theo Slide 79, không nuốt exception hoặc ngụy tạo số liệu.

## Output directory đề xuất (Slide 76)

```text
artifacts/
  data_audit.json
  splits/
    split_run_1.json
    split_run_2.json
    ...
  checkpoints/
    h1_model_run_1.pt
    ...
  graphs/
    b0_domain_dag.png
    h2_adjacency_run_1.png
    h2_learned_dag_run_1.png
  results/
    raw_paired_runs.csv
    prediction_scores.csv
    intervention_estimates.csv
    refutations.json
  tables/
    table_a_prediction.csv
    table_b_interventions.csv
    table_c_robustness_cost.csv
    optional_table_synthetic.csv
    optional_table_ablations.csv
```

## Bộ 3 Bảng Bắt buộc (Slide 74–75)

**Bảng A - Standard Prediction Comparison (Slide 74):**
- Đánh giá trên cùng tập $Y_{\text{test}}$ qua $\ge 5$ paired runs.
- Các cột: Model, Run ID / Summary (Mean $\pm$ Std), Graph ID, AUROC, Macro-F1, Brier Score (hoặc Log Loss), $\Delta_{\text{Model}-B_0}$ (AUROC), $\Delta_{\text{Model}-B_0}$ (F1), Fit Time (s), Predict Time (s), Hardware.

**Bảng B - Interventions & Standardized Causal Effects (Slide 75):**
- Chuẩn hóa trên cùng tập tham chiếu kiểm thử $C_{\text{test}}$ qua $\ge 5$ paired runs.
- Các cột: Model, Summary (Mean $\pm$ Std), Quần thể tham chiếu & Cỡ mẫu $n_{\text{test}}$, Công thức định danh & Tập điều chỉnh, $\text{risk}(T=0)$, $\text{risk}(T=1)$, $\widehat{\text{ATE}}$, 95% Confidence Interval, Phương pháp tính CI, Trạng thái Overlap / Đồ thị.

**Bảng C - Robustness and Computational Cost (Slide 75):**
- Đánh giá tính vững qua DoWhy refuters và chi phí tính toán qua $\ge 5$ paired runs.
- Các cột: Model, Hiệu ứng Placebo Treatment Refuter, Hiệu ứng Data Subset Refuter (80% subset), Độ tản mạn hiệu ứng 5 runs ($\text{Max} - \text{Min}$), $\Delta_{\text{ATE}}(\text{Model}-B_0)$, Tổng thời gian thực thi (Fit + Query runtime), Cấu hình phần cứng.

*(Các kết quả mở rộng tùy chọn như Synthetic Tutoring Benchmark ATE=0.30 hoặc Ablations A1–A3 được lưu riêng vào các file `optional_table_*.csv` và đưa vào Phụ lục).*

## Điều kiện chấp nhận trước khi nộp (Slide 76, 79)

- `Restart runtime and Run all` trên Google Colab kernel mới tinh không phát sinh bất kỳ lỗi nào.
- Toàn bộ kết quả trong bảng và đồ thị đều truy vết được về seed, split và checkpoint tương ứng.
- Không có kết quả placeholder `...`, số 0 giả hoặc fallback che đậy lỗi.
- Có ghi log rõ ràng cách xử lý các trường hợp thất bại (ô thưa, chu trình H2, lỗi định danh) theo Slide 79.
- Hiển thị tường minh sự khác biệt giữa quan sát $P(Y \mid T)$ và can thiệp $P(Y \mid \text{do}(T))$.
- Có file môi trường phụ thuộc (`requirements.txt`) ghi rõ phiên bản thư viện.
- Báo cáo trung thực, khách quan; không kết luận thiên lệch “mô hình nơ-ron luôn tốt hơn”.
