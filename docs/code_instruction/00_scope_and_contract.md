# Phạm vi, biến số và hợp đồng đầu ra

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

## 3. Experiment contract

Mỗi lần chạy phải lưu:

```text
seed, model, split_id, n_train, n_val, n_test,
p_do_t0, p_do_t1, ate, ate_ci_low, ate_ci_high,
auroc, macro_f1, brier_score, runtime_seconds, status, error
```

Các trạng thái lỗi phải hiện rõ (`failed`, thông báo lỗi và model/seed liên
quan), không được thay bằng số 0 hoặc kết quả giả.

## 4. Công thức bắt buộc

```text
p_do(t) = sum_z P(Y=1 | T=t, Z=z) P(Z=z)
ATE     = p_do(1) - p_do(0)
```

Luôn in cạnh nhau:

- `P(Y=1 | T=t)` để minh họa quan sát;
- `P(Y=1 | do(T=t))` để kết luận can thiệp.

