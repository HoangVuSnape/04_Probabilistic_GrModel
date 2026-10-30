# Data pipeline cần viết

## 1. Nạp dữ liệu

Ưu tiên `ucimlrepo` để lấy UCI ID 697. Có thể cho phép đường dẫn local
fallback, nhưng phải ghi rõ nguồn trong output và không âm thầm dùng dữ liệu
khác.

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=697)
raw = dataset.data.features.copy()
raw["Target"] = dataset.data.targets["Target"]
```

## 2. Làm sạch và lọc cohort

1. Chuẩn hóa tên cột, giữ bản sao raw bất biến.
2. Kiểm tra `Target` có đúng ba trạng thái.
3. Lọc `Target != "Enrolled"`.
4. Ánh xạ `Dropout=0`, `Graduate=1`.
5. Kiểm tra không còn missing ở các cột sử dụng.

## 3. Tạo biến mô hình

Các biến baseline phải được tạo bằng hàm thuần, có test:

```python
def make_model_frame(raw):
    out = raw[[
        "Scholarship holder", "Admission grade", "Debtor",
        "Tuition fees up to date", "Mother's qualification",
        "Age at enrollment", "Target"
    ]].copy()
    out["Z1"] = pd.cut(
        out.pop("Admission grade"),
        bins=[-float("inf"), 120, 140, float("inf")],
        labels=[0, 1, 2],
    ).astype(int)
    out["Z2"] = out.pop("Debtor").astype(int)
    out["Z3"] = out.pop("Tuition fees up to date").astype(int)
    out["Z4"] = (out.pop("Mother's qualification") >= 13).astype(int)
    out["Z5"] = (out.pop("Age at enrollment") > 20).astype(int)
    out["T"] = out.pop("Scholarship holder").astype(int)
    out["Y"] = out.pop("Target").map({"Dropout": 0, "Graduate": 1})
    return out[["Z1", "Z2", "Z3", "Z4", "Z5", "T", "Y"]]
```

Ngưỡng `Z4` phải được xác nhận lại từ data dictionary trước khi chốt; không
được giả định ý nghĩa mã số nếu chưa kiểm tra metadata UCI.

## 4. Split công bằng

Với mỗi seed `[42, 123, 456, 789, 999]`:

- split stratified 70/15/15;
- stratify ít nhất theo `Y`, và kiểm tra phân bố `T`;
- fit discretizer/encoder chỉ trên train, rồi transform validation/test;
- lưu index train/val/test thành `artifacts/splits/seed_<seed>.json`.

## 5. Kiểm tra trước khi train

- `Y` chỉ có `{0, 1}`.
- `T` chỉ có `{0, 1}`.
- mọi cấu hình dùng có cả `T=0` và `T=1`;
- không có ô điều chỉnh cần thiết bị rỗng mà không được cảnh báo;
- không sử dụng `Curricular units ...` hoặc điểm học kỳ sau học bổng làm
  baseline confounder.
