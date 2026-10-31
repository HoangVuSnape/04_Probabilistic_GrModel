# Phạm vi, biến số và hợp đồng đầu ra
> Cập nhật theo Slide 51, 54, 55, 61, 74–75, 76 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

## 0. Quy chuẩn phong cách code

Code phải giữ phong cách academic, non-OOP và có thể truy vết từ input đến
output. Mỗi hàm nên làm một việc, nhận dependency qua tham số và trả về kết
quả rõ ràng; tránh đọc/ghi state ẩn từ notebook.

Dùng dataclass cho các record có cấu trúc cố định:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    artifact_dir: Path


@dataclass(frozen=True)
class ExperimentResult:
    seed: int
    model: str
    status: str
    p_do_t0: float | None
    p_do_t1: float | None
    ate: float | None
    error: str | None = None
```

`ExperimentConfig` được tạo ở cell setup và truyền vào các hàm. Không tạo
`GLOBAL_CONFIG`, `GLOBAL_MODEL`, `GLOBAL_DF` hoặc biến toàn cục tương tự.

## 1. Causal contract

| Ký hiệu | Cột dữ liệu | Vai trò | Mã hóa |
|---|---|---|---|
| `T` | `Scholarship holder` | treatment | `0/1` |
| `Y` | `Target` | outcome | `Dropout -> 0`, `Graduate -> 1` |
| `Z1` | `Admission grade` | baseline confounder | 3 nhóm |
| `Z2` | `Debtor` | baseline confounder | `0/1` |
| `Z3` | `Tuition fees up to date` | baseline confounder | `0/1` |
| `Z4` | `Mother's qualification` | baseline confounder | 2 nhóm |
| `Z5` | `Age at enrollment` | baseline confounder | 2 nhóm |

DAG B0:

```text
Z1..Z5 -> T -> Y
Z1..Z5 --------> Y
```

Tập điều chỉnh backdoor mặc định là `Z = {Z1, Z2, Z3, Z4, Z5}`.

## 2. Dataset contract

- Nguồn: UCI Machine Learning Repository, ID 697.
- Dữ liệu thô: 4,424 dòng.
- Loại 794 dòng `Target == "Enrolled"`.
- Cohort cuối: 3,630 dòng, chỉ còn `Graduate` và `Dropout`.
- Kiểm tra missing, duplicate, domain/range và treatment overlap trước khi
  train.

Primary execution dùng đúng năm biến `Z1..Z5` và các cạnh B0 trong tài liệu
quyết định. Các biến nhập học bổ sung chỉ chạy ở sensitivity analysis sau khi
đã kiểm tra sparse cells và positivity; các biến curricular units học kỳ 1/2
không được đưa vào admission-time adjustment.

## 3. Experiment contract

Mỗi lần chạy phải lưu:

```text
seed, model, split_id, n_train, n_val, n_test,
p_do_t0, p_do_t1, ate, ate_ci_low, ate_ci_high,
auroc, macro_f1, brier_score, runtime_seconds, status, error
```

Các trạng thái lỗi phải hiện rõ (`failed`, thông báo lỗi và model/seed liên
quan), không được thay bằng số 0 hoặc kết quả giả.

## 4. Công thức bắt buộc (Slide 61)

```text
q_m(t, Z_i) = P_m(Y=1 | T=t, Z_i)
p_do(t)     = risk_m(t) = (1 / n_test) * sum_i q_m(t, Z_i)  [tương đương sum_z P(Y=1 | T=t, Z=z) * P(Z=z)]
ATE         = p_do(1) - p_do(0)
```

> **Lưu ý sống còn:** Nhãn kết quả `Y_test` tuyệt đối không được dùng trong tính toán `risk_m(t)` và `ATE`.
> `Y_test` chỉ dùng để chấm điểm dự báo ngoài mẫu (AUROC, Macro-F1, Brier Score).

Luôn in cạnh nhau:

- `P(Y=1 | T=t)` để minh họa quan sát;
- `P(Y=1 | do(T=t))` để kết luận can thiệp.
