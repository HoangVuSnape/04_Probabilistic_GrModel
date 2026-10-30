# BÁO CÁO KHẢO SÁT DỮ LIỆU (EDA)
## UCI 697 và Kaggle Playground Series S4E6

> **Notebook thực thi:** [eda_uci697_kaggle_s4e6.ipynb](../codeCite/eda_uci697_kaggle_s4e6.ipynb)  
> **Ngày tạo:** 2026-10-29  
> **Trạng thái:** Đã tạo pipeline EDA; cần chạy trong Python/Jupyter/Colab để điền các con số và artifacts thực tế.

---

## 1. Mục tiêu

Trước khi xây dựng B0/H1/H2, cần khảo sát dữ liệu như một bài phân tích dữ liệu độc lập:

1. Kiểm tra schema, kiểu dữ liệu, missing values và duplicate.
2. Khảo sát phân bố treatment `Scholarship holder` và outcome `Target`.
3. Kiểm tra treatment overlap theo các nhóm baseline.
4. Vẽ các biểu đồ phân phối, countplot, boxplot, mutual information và correlation heatmap.
5. Phân biệt association dùng cho EDA với causal effect dùng cho intervention.
6. So sánh UCI gốc với Kaggle synthetic.
7. Xác định biến nào có thể dùng trong admission-time setting và biến nào gây temporal leakage.
8. Đánh giá mức độ sẵn sàng cho causal analysis trước khi chốt DAG.

Notebook được thiết kế theo tinh thần đoạn code mẫu `G3/studytime/higher`, nhưng chuyển sang các biến có ý nghĩa đối với UCI 697.

---

## 2. Cách chạy

### UCI 697

Notebook tự tải dữ liệu từ URL chính thức:

```text
https://archive.ics.uci.edu/static/public/697/data.csv
```

### Kaggle S4E6

Do Kaggle API yêu cầu xác thực, tải thủ công `train.csv` rồi đặt vào:

```text
data/kaggle_s4e6/train.csv
```

Notebook sẽ tự dò thêm các vị trí:

- `data/train.csv`
- `/content/data/kaggle_s4e6/train.csv`
- `/kaggle/input/playground-series-s4e6/train.csv`

Nếu chưa có Kaggle CSV, notebook vẫn chạy toàn bộ phần UCI và thông báo rõ phần so sánh bị bỏ qua.

Các hình được lưu vào:

```text
artifacts/eda/
```

---

## 3. Các phân tích và biểu đồ

### 3.1. Data audit

Notebook xuất:

- số dòng và số cột;
- kiểu dữ liệu;
- số giá trị unique;
- missing cells;
- duplicate rows;
- duplicate sau khi bỏ `id`;
- phân bố `Target`.

Đây là bước kiểm tra chất lượng, chưa phải bước chọn mô hình.

### 3.2. Treatment/outcome audit

Các bảng được tạo:

- phân bố `Scholarship holder`;
- phân bố `Target`;
- bảng chéo treatment × outcome;
- tỷ lệ treatment theo tertile của `Admission grade`.

Bảng overlap không được dùng để kết luận học bổng có tác động. Nó chỉ trả lời liệu mỗi nhóm baseline có đủ cả treatment và control để tiếp tục suy luận hay không.

### 3.3. Bộ biểu đồ chính

Notebook tạo cho từng dataset:

1. Histogram/KDE của `Admission grade`.
2. Countplot số mẫu theo `Scholarship holder`.
3. Boxplot `Admission grade` theo nhóm tuổi và `Target`.
4. Profile treatment: điểm tuyển sinh, tuổi, nợ học phí, đóng học phí đúng hạn.
5. Top 15 feature theo Mutual Information.
6. Correlation heatmap.
7. Khi có Kaggle: so sánh phân phối UCI/Kaggle và tỷ lệ `Target`.

### 3.4. Phân tích nâng cao

Notebook đã bổ sung thêm một section chuyên sâu:

- Bảng missingness và cardinality của từng biến.
- Class balance của `Target`.
- Boxplot tổng quan outlier cho các biến số.
- Tỷ lệ outcome theo treatment.
- Kiểm tra số lượng biến baseline và biến hậu nhập học.
- Treatment overlap theo đồng thời nhóm `Admission grade` và nhóm tuổi.
- Observed rate difference theo từng tầng để phát hiện dị biệt giữa các nhóm.
- Top cặp biến số tương quan cao để kiểm tra redundancy.
- Lưu bảng stratified observed difference vào artifacts.

Các observed rate difference chỉ là mô tả phân tầng. Chúng không thay thế ATE sau Backdoor adjustment.

Đây là phiên bản tương ứng với:

```python
sns.histplot(...)
sns.countplot(...)
sns.boxplot(...)
mutual_info_...
sns.heatmap(...)
```

trong code tham khảo của người dùng.

---

## 4. Diễn giải cần dùng trong báo cáo

### 4.1. Mutual Information

Mutual Information có thể dùng để:

- tìm feature có liên hệ thông tin với `Target`;
- ưu tiên biến để khảo sát;
- phát hiện các biến học kỳ có sức dự báo rất mạnh;
- kiểm tra distribution shift giữa UCI và Kaggle.

Không được viết:

> “Mutual Information cao chứng minh biến đó gây ra tốt nghiệp.”

Cách viết đúng:

> “Mutual Information cho thấy feature có phụ thuộc thông tin với outcome; kết quả này được dùng để khảo sát và thiết kế mô hình, không phải bằng chứng về hướng nhân quả.”

### 4.2. Correlation heatmap

Correlation giúp phát hiện:

- biến trùng lặp;
- nhóm biến có phụ thuộc mạnh;
- khả năng collinearity;
- biến hậu kỳ có liên hệ gần như trực tiếp với outcome.

Các biến `Curricular units ...` và điểm học kỳ cần được đánh dấu riêng. Chúng phù hợp với `semester-aware prediction`, nhưng không được đưa vào admission-time prediction nếu chưa quan sát tại thời điểm nhập học.

### 4.3. Boxplot và bảng profile

Boxplot cho phép mô tả khác biệt quan sát giữa nhóm treatment/outcome. Ví dụ, nếu nhóm nhận học bổng có phân bố `Admission grade` khác nhóm không nhận, đây là dấu hiệu cần điều chỉnh confounding, không phải bằng chứng học bổng gây ra outcome.

---

## 5. Kế hoạch phân tích theo hai dataset

### UCI 697

UCI là nguồn chính cho:

- data audit;
- định nghĩa cohort `Dropout`/`Graduate`;
- lựa chọn baseline covariates;
- kiểm tra overlap;
- Bảng A causal estimation.

### Kaggle S4E6

Kaggle là nguồn phụ cho:

- benchmark synthetic;
- kiểm tra robustness;
- so sánh phân phối;
- kiểm tra transfer UCI → Kaggle và Kaggle → UCI sau khi pipeline chính hoàn tất.

Không được mô tả Kaggle là khảo sát của hơn 76.000 sinh viên thật.

---

## 6. Quy tắc quyết định feature sau EDA

### Admission-time feature set

Chỉ giữ các biến có thể biết trước hoặc tại thời điểm nhập học, sau khi xác minh thời điểm đo:

- `Admission grade`;
- `Age at enrollment`;
- `Course`;
- `Application mode`;
- `Application order`;
- `Previous qualification`;
- học vấn cha mẹ;
- `Scholarship holder`;
- `Debtor`;
- `Tuition fees up to date`;
- biến nhân khẩu học/kinh tế - xã hội phù hợp.

### Semester-aware feature set

Có thể thêm biến học kỳ 1 hoặc học kỳ 2, nhưng phải ghi rõ đây là dự báo cập nhật trong quá trình học.

### Không dùng

- `id`;
- biến được xác định là hậu treatment trong Backdoor adjustment;
- biến học kỳ khi đang tuyên bố là admission-time prediction.

---

## 7. Output cần thu sau khi chạy notebook

Sau khi chạy trong Colab/Jupyter, cần lưu lại:

```text
artifacts/eda/
├── uci_core_eda.png
├── uci_mutual_information.png
├── uci_correlation_heatmap.png
├── kaggle_core_eda.png
├── kaggle_mutual_information.png
├── kaggle_correlation_heatmap.png
├── uci_kaggle_comparison.png
└── uci_kaggle_numeric_comparison.csv
```

Tên file Kaggle chỉ xuất hiện nếu `train.csv` được cung cấp.

---

## 8. Hạn chế hiện tại

Trong workspace hiện tại chưa có CSV UCI/Kaggle và môi trường Python/Jupyter chưa khả dụng để chạy notebook trực tiếp. Vì vậy, báo cáo này không tự điền các mean, tỷ lệ hoặc MI score giả định. Notebook sẽ tải UCI và in số liệu thật khi chạy trong Colab/Jupyter.

Đây là lựa chọn có chủ ý để tránh tạo số liệu không thể truy nguyên. Sau khi chạy, các giá trị trong report Final phải lấy từ output/artifacts của notebook, không nhập tay.

---

## 9. Kết luận EDA cần hướng tới

EDA phải trả lời được bốn câu hỏi trước khi chuyển sang causal modeling:

1. Treatment có đủ overlap trong các nhóm baseline không?
2. Outcome có bị mất cân bằng hoặc bị ảnh hưởng bởi cách xử lý `Enrolled` không?
3. Feature nào là baseline, feature nào là hậu kỳ/leakage?
4. Kaggle synthetic có phân phối đủ gần UCI để dùng làm robustness benchmark không?

Chỉ sau khi bốn câu hỏi này được trả lời bằng output thực tế mới chốt tập $\mathbf{Z}$, DAG B0 và pipeline B0/H1/H2.
