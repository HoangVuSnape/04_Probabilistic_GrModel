# Hướng dẫn triển khai mã nguồn

Tài liệu này chuyển yêu cầu trong slide môn IT701100 thành kế hoạch code có thể
triển khai trong Google Colab/Jupyter. Nguồn tham chiếu chính là:

- `..\pdf\HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`
- `..\requirements\midterm\README.md`
- `..\requirements\final\README.md`
- `..\..\research\01_problem_formulation.md`
- `..\..\research\02_dataset_specification.md`
- `..\..\report\decisions\ADR_001_dataset_selection.md`

## Mục tiêu cuối cùng

Notebook phải trả lời được câu hỏi:

> Nếu can thiệp cấp học bổng cho sinh viên (`do(T=1)`) thay vì không cấp
> (`do(T=0)`), xác suất tốt nghiệp (`Y=1`) thay đổi bao nhiêu?

Không được dùng `P(Y | T=t)` để kết luận hiệu quả chính sách. Kết quả chính
phải dùng `P(Y | do(T=t))`, ATE, rủi ro và độ bất định.

## Thứ tự triển khai

1. Đọc [00_scope_and_contract.md](./00_scope_and_contract.md).
2. Viết pipeline dữ liệu theo [01_data_pipeline.md](./01_data_pipeline.md).
3. Viết synthetic benchmark trước để kiểm tra do-operator.
4. Cài B0, sau đó H1, H2 và A1-A3 theo
   [02_models_and_ablations.md](./02_models_and_ablations.md).
5. Ghép tất cả thành notebook theo
   [03_notebook_cell_order.md](./03_notebook_cell_order.md).
6. Chạy checklist trong [04_validation_and_outputs.md](./04_validation_and_outputs.md).

## Phạm vi bắt buộc và tùy chọn

| Hạng mục | Trạng thái |
|---|---|
| Synthetic benchmark, true ATE = 0.30 | Bắt buộc |
| UCI ID 697, loại `Enrolled` | Bắt buộc |
| B0, H1, H2 | Bắt buộc |
| A1, A2, A3, one-change-only | Bắt buộc |
| 5 paired seeds: `42, 123, 456, 789, 999` | Bắt buộc |
| DoWhy identify/estimate + 2 refuters | Bắt buộc |
| DoWhy GCM intervention | Bắt buộc |
| H1+H2 | Tùy chọn sau khi core chạy ổn định |

## Quy tắc quan trọng

- Tách rõ discovery, identification và estimation.
- Chỉ fit preprocessing trên train của từng seed.
- Không đưa điểm học kỳ 1/2 vào các biến nền của causal question; đây là
  biến sau can thiệp hoặc có nguy cơ leakage.
- Mọi model dùng cùng split và cùng test set trong một seed.
- Không gọi một đồ thị do LLM gợi ý là “neural structure learning”.
- H2 chỉ được xem là thành công khi có tối ưu liên tục, DAG constraint và
  domain mask được kiểm tra bằng số liệu.
- Lưu cả kết quả thành `.csv`/`.json`, không chỉ in ra notebook.

