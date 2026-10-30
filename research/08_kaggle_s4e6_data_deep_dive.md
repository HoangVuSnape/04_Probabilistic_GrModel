# ĐÀO SÂU KAGGLE PLAYGROUND SERIES S4E6
## Synthetic Academic Success Dataset và mối liên hệ với UCI 697

> **Ngày rà soát:** 2026-10-29  
> **Mục đích:** Tài liệu tham khảo cho việc quyết định có dùng Kaggle S4E6 trong benchmark hay không.  
> **Kết luận chính:** Kaggle S4E6 phù hợp làm benchmark dự báo tabular; không nên xem là dữ liệu quan sát của hơn 76.000 sinh viên thật và không nên dùng riêng nó để khẳng định hiệu ứng nhân quả của học bổng.

---

## 1. Tóm tắt điều hành

Kaggle Playground Series - Season 4, Episode 6 (S4E6) có tên **Classification with an Academic Success Dataset**. Kaggle xác nhận `train.csv` và `test.csv` được tạo bởi một mô hình deep learning được huấn luyện trên bộ UCI **Predict Students' Dropout and Academic Success** (UCI ID 697). Kaggle cũng ghi rõ các phân phối feature gần với, nhưng không hoàn toàn giống, dữ liệu gốc.

Vì vậy, đây là một **synthetic benchmark có nguồn gốc từ dữ liệu thực**, không phải một cuộc điều tra mới với hàng chục nghìn sinh viên. Kích thước lớn giúp chạy benchmark Machine Learning ổn định hơn, nhưng không tự động tạo thêm số lượng cá thể độc lập hay tăng khả năng suy rộng sang trường đại học khác.

Đối với đề tài Causal Bayesian Network của nhóm:

1. Dùng **UCI 697 làm dữ liệu chính** cho lập luận bối cảnh và phân tích nhân quả.
2. Dùng **Kaggle S4E6 làm benchmark/robustness dataset**.
3. Báo cáo riêng hai setting: dự báo tại thời điểm nhập học và dự báo sau khi đã quan sát kết quả học kỳ.
4. Không diễn giải accuracy trên Kaggle thành hiệu quả của chính sách học bổng.

---

## 2. Các nguồn đã kiểm tra

### 2.1. Nguồn chính thức

| Nguồn | Nội dung cần kiểm tra | Liên kết |
|---|---|---|
| Kaggle Overview | Mục tiêu, metric, thời gian thi, bản chất Tabular Playground | [Competition Overview](https://www.kaggle.com/competitions/playground-series-s4e6) |
| Kaggle Data | Mô tả synthetic generator, file train/test, liên kết UCI | [Competition Data](https://www.kaggle.com/competitions/playground-series-s4e6/data) |
| Kaggle Discussion | Ý kiến cộng đồng, chiến lược mô hình, write-up | [Discussion](https://www.kaggle.com/competitions/playground-series-s4e6/discussion) |
| UCI metadata | Kích thước, biến, nhãn, nguồn gốc và bài báo | [UCI Dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| UCI API | Metadata máy đọc được và DOI | [UCI API](https://archive.ics.uci.edu/api/dataset?id=697) |
| Bài báo UCI | Nguồn học thuật giới thiệu bài toán | [Early prediction of student's performance](https://doi.org/10.1007/978-3-030-72657-7_16) |

### 2.2. Một số discussion/write-up đáng đọc

Các trang dưới đây phản ánh cách cộng đồng Kaggle giải bài toán. Chúng hữu ích để hiểu hiệu năng mô hình và thực hành benchmark, nhưng không thay thế cho kiểm định khoa học về tính độc lập, tính đại diện hoặc quan hệ nhân quả.

- [3rd place solution: a single xgb model](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/515983)
- [1st Place solution was ....](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/516282)
- [For people searching good features with better MI scores](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/515568)
- [Finding out whether two models in this competition are likely to ensemble well](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/512220)
- [The average jump upward in the top 10 was 517 spots](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/515966)
- [Top118 solution](https://www.kaggle.com/competitions/playground-series-s4e6/writeups/lgb-is-enough-top118-solution)
- [10th Place Solution](https://www.kaggle.com/competitions/playground-series-s4e6/writeups/lash-fire-10th-place-solution)

---

## 3. Kaggle S4E6 thực sự là dữ liệu gì?

### 3.1. Không phải bản mở rộng trực tiếp của UCI

UCI 697 có:

- 4.424 quan sát;
- 36 feature;
- không có missing value;
- ba nhãn `Dropout`, `Enrolled`, `Graduate`;
- dữ liệu từ một cơ sở giáo dục đại học ở Bồ Đào Nha.

Kaggle S4E6 sinh các dòng mới từ phân phối học được trên UCI. Do đó, câu mô tả chính xác là:

> “Kaggle sinh một tập dữ liệu tổng hợp có khoảng 76.000 dòng train từ phân phối được học bằng deep learning trên UCI 697.”

Không nên viết:

> “Kaggle thu thập dữ liệu của hơn 76.000 sinh viên.”

Trang Kaggle hiển thị ví dụ submission bắt đầu từ ID `76518`, phù hợp với cách hiểu rằng train có khoảng 76.518 dòng và test bắt đầu sau vùng ID đó. Khi báo cáo số dòng chính thức, vẫn cần kiểm tra lại trực tiếp từ file CSV đang dùng.

### 3.2. Mục tiêu của Kaggle khác mục tiêu nhân quả

Kaggle đặt bài toán dự báo nhãn `Target` và đánh giá bằng **accuracy**. Mục tiêu của nhóm lại là ước lượng:

$$
\mathrm{ATE}
= P(Y=1 \mid do(T=1))
- P(Y=1 \mid do(T=0)).
$$

Hai mục tiêu này không tương đương:

- Accuracy trả lời mô hình dự đoán đúng nhãn đến mức nào.
- ATE trả lời outcome thay đổi thế nào dưới một can thiệp giả định.

Một mô hình có accuracy cao vẫn có thể:

- học correlation thay vì causal effect;
- sử dụng biến xảy ra sau treatment;
- dự báo tốt nhưng không nhận diện được hiệu ứng học bổng;
- không chuyển được sang một trường đại học khác.

---

## 4. Điểm mạnh của bộ Kaggle

### 4.1. Quy mô vật lý lớn

Tập train lớn hơn UCI nhiều lần, thuận tiện cho:

- benchmark nhiều mô hình;
- chạy nhiều random seeds;
- huấn luyện LightGBM, XGBoost, CatBoost và neural network;
- kiểm tra feature engineering;
- giảm tình trạng ô rỗng khi chỉ xét benchmark dự báo.

### 4.2. Feature vẫn có ngữ nghĩa miền

Các biến như `Admission grade`, `Age at enrollment`, `Scholarship holder`, `Debtor`, `Tuition fees up to date`, `Course` và điểm học kỳ vẫn giúp bài toán giữ được ngữ cảnh giáo dục thực tế.

### 4.3. Tốt cho thực hành mô hình tabular

Discussion của Kaggle cho thấy các hướng tiếp cận phổ biến gồm:

- mô hình boosting;
- XGBoost/LightGBM;
- feature selection và mutual information;
- ensemble;
- theo dõi public/private leaderboard;
- validation và tuning.

Đây là môi trường tốt để so sánh năng lực dự báo của B0/H1/H2, miễn là kết quả được gắn nhãn đúng là benchmark synthetic.

### 4.4. Có thể làm kiểm tra độ bền

Có thể dùng Kaggle để kiểm tra:

- mô hình học trên UCI có giữ được hiệu năng trên synthetic distribution hay không;
- mô hình học trên Kaggle có chuyển ngược về UCI hay không;
- cấu trúc DAG và CPD có nhạy với thay đổi phân phối không.

---

## 5. Hạn chế và tranh luận cần ghi rõ

### 5.1. “Nhiều dòng” không đồng nghĩa “nhiều thông tin độc lập”

Dữ liệu synthetic được sinh từ phân phối học trên 4.424 quan sát. Các dòng mới có thể:

- giống nhau hoặc gần giống nhau;
- tái tạo các pattern của dữ liệu gốc;
- không đại diện cho các cá thể độc lập theo nghĩa khảo sát;
- làm cho sai số chuẩn có vẻ nhỏ hơn mức đáng tin trong quần thể thực.

Vì vậy, không nên dùng `N = 76.518` để suy ra rằng độ chắc chắn thống kê tương đương khảo sát 76.518 sinh viên thật.

### 5.2. Có thể tồn tại artifact của generator

Kaggle công khai rằng dữ liệu synthetic có thể còn những thách thức và artifact. Generator có thể tái tạo tốt phân phối biên nhưng không bảo toàn hoàn hảo:

- phụ thuộc có điều kiện;
- các kết hợp hiếm;
- quan hệ phi tuyến;
- cấu trúc thời gian;
- cơ chế tạo nhãn;
- quan hệ nhân quả.

Các artifact này có thể giúp hoặc làm hại một mô hình tùy theo cách chia dữ liệu và metric.

### 5.3. Không thể suy ra causal effect chỉ từ dữ liệu synthetic

Nếu `Scholarship holder` tương quan với `Graduate`, điều đó chưa chứng minh học bổng gây ra tốt nghiệp. Có thể còn các yếu tố:

- năng lực học tập chưa đo đủ;
- điều kiện gia đình;
- động lực cá nhân;
- chính sách xét học bổng;
- sức khỏe hoặc hoàn cảnh xã hội;
- khác biệt giữa ngành học.

Backdoor adjustment chỉ có ý nghĩa khi DAG và giả định không còn nhiễu chưa quan sát được bảo vệ hợp lý. Synthetic generator không cung cấp bằng chứng rằng các giả định causal đó đúng.

### 5.4. Giới hạn suy rộng

UCI gốc đến từ một cơ sở giáo dục cụ thể ở Bồ Đào Nha. Kaggle synthetic kế thừa giới hạn này. Không nên suy rộng trực tiếp sang:

- đại học Việt Nam;
- trường tư thục hoặc hệ đào tạo khác;
- quốc gia khác;
- giai đoạn thời gian khác;
- chính sách học bổng khác.

### 5.5. Leaderboard có thể bị overfit

Discussion về thay đổi public/private rank cho thấy điểm public leaderboard không phải đánh giá hoàn toàn ổn định. Cần tránh:

- chọn mô hình chỉ vì một điểm public;
- thử quá nhiều lần trên cùng một leaderboard;
- xem leaderboard là external validation;
- bỏ qua nhiều-seed validation.

---

## 6. Rủi ro temporal leakage

UCI 697 trộn các biến biết được ở những thời điểm khác nhau.

### 6.1. Biến có thể dùng trong admission-time setting

Ví dụ:

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
- biến nhân khẩu học và kinh tế - xã hội phù hợp.

### 6.2. Biến phát sinh sau khi sinh viên bắt đầu học

Ví dụ:

- `Curricular units 1st sem (approved)`;
- `Curricular units 1st sem (grade)`;
- `Curricular units 2nd sem (approved)`;
- `Curricular units 2nd sem (grade)`;
- các biến thi, đăng ký và kết quả học kỳ.

Nếu dùng các biến này nhưng gọi kết quả là “dự báo nguy cơ tại thời điểm nhập học”, đó là leakage. Đặc biệt, điểm học kỳ 2 có thể gần như là mediator hoặc proxy rất mạnh cho kết quả cuối khóa.

### 6.3. Nên báo cáo hai setting

| Setting | Feature | Diễn giải |
|---|---|---|
| Admission-time | Chỉ biến đã biết trước/tại thời điểm nhập học | Dự báo sớm |
| Semester-aware | Thêm biến học kỳ 1 hoặc 2 | Dự báo cập nhật trong quá trình học |

Không gộp hai kết quả này vào một bảng mà không ghi rõ thời điểm quan sát.

---

## 7. Khả năng leakage giữa UCI và Kaggle

Kaggle công khai UCI gốc và còn cho phép người thi dùng UCI để khám phá hoặc cải thiện điểm. Điều này thuận tiện cho nghiên cứu nhưng phải kiểm soát:

1. Exact duplicate sau khi bỏ cột `id`.
2. Duplicate giữa UCI và Kaggle train/test.
3. Near-duplicate bằng Hamming distance hoặc Gower distance.
4. Khác biệt phân phối giữa UCI, Kaggle train và Kaggle test.
5. Việc gộp UCI vào train có làm thay đổi mục tiêu benchmark hay không.

Nên phân biệt rõ:

- **External transfer:** train một nguồn, test nguồn còn lại.
- **Pooled training:** gộp hai nguồn vào train.
- **Synthetic augmentation:** dùng UCI để bổ sung cho Kaggle.

Ba thiết kế này không được báo cáo như cùng một thí nghiệm.

---

## 8. Khuyến nghị thiết kế thực nghiệm cho đề tài

### 8.1. Vai trò của từng dataset

| Dataset | Vai trò khuyến nghị |
|---|---|
| UCI 697 | Dataset chính cho lập luận khoa học và causal analysis |
| Kaggle S4E6 | Synthetic benchmark và robustness check |
| UCI + Kaggle | Chỉ dùng trong thí nghiệm augmentation/transfer có ghi rõ |

### 8.2. Ba nhóm thí nghiệm

**Nhóm A - UCI original**

- Báo cáo kết quả trên dữ liệu gốc.
- Giữ hoặc loại `Enrolled` theo câu hỏi nghiên cứu.
- Ghi rõ quy tắc lọc.

**Nhóm B - Kaggle synthetic**

- Dùng cho accuracy/AUROC/Macro-F1/Brier và runtime.
- Không gọi là 76.000 sinh viên thật.
- Kiểm tra duplicate và distribution shift.

**Nhóm C - Cross-source transfer**

- Train UCI, test Kaggle.
- Train Kaggle, test UCI.
- So sánh với random split trong cùng nguồn.

Nếu cross-source performance giảm mạnh, cần xem đó là bằng chứng về distribution shift, không chỉ là “mô hình kém”.

### 8.3. Các kiểm tra tối thiểu

```text
1. Đọc số dòng và schema thật từ file CSV đang dùng.
2. Xác định cột id có thông tin hay chỉ là định danh.
3. So sánh phân phối từng feature giữa UCI và Kaggle.
4. So sánh phân bố Target.
5. Kiểm tra exact duplicate sau khi bỏ id.
6. Kiểm tra near-duplicate.
7. Chạy tối thiểu 5 seeds.
8. Tách admission-time và semester-aware.
9. Đánh giá trên tập ngoài nguồn nếu có thể.
10. Không diễn giải accuracy thành ATE.
```

---

## 9. Kết luận sử dụng cho báo cáo

Đoạn mô tả có thể sử dụng sau khi nhóm tự đối chiếu với kết quả chạy thực tế:

> Kaggle Playground Series S4E6 là một benchmark phân loại tabular tổng hợp dựa trên UCI 697. Dữ liệu có lợi thế về kích thước vật lý, khả năng tái lập và tính thuận tiện cho việc so sánh các mô hình Machine Learning hiện đại. Tuy nhiên, các dòng synthetic không nên được diễn giải như các sinh viên độc lập có thật; generator có thể tạo artifact và không bảo đảm bảo toàn cấu trúc nhân quả. Do đó, trong nghiên cứu này, Kaggle S4E6 được dùng cho benchmark dự báo và phân tích độ bền, trong khi các kết luận về hiệu ứng can thiệp của học bổng được gắn với UCI gốc, giả định DAG, kiểm soát temporal leakage và phân tích độ nhạy.

### Quy tắc diễn giải cuối cùng

- Được nói: **“76.518 dòng synthetic được sinh từ UCI 697.”**
- Không nên nói: **“76.518 sinh viên được khảo sát.”**
- Được nói: **“mô hình đạt accuracy trên Kaggle benchmark.”**
- Không nên nói: **“học bổng làm tăng xác suất tốt nghiệp vì accuracy cao.”**
- Được nói: **“Kaggle hỗ trợ kiểm tra độ bền của mô hình.”**
- Không nên nói: **“Kaggle chứng minh quan hệ nhân quả.”**

