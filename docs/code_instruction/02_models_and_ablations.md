# Các mô hình phải cài đặt (Core Models & Optional Extensions)
> Cập nhật theo Slide 51, 53, 57–63, 64–67, 75, 80 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

## Chuẩn tổ chức code

Không viết B0/H1/H2 thành các class model phức tạp nếu không có yêu cầu từ
framework. Tổ chức bằng các hàm academic, ví dụ:

```python
def fit_b0(train_frame: pd.DataFrame, config: ExperimentConfig):
    ...


def estimate_do_effect(
    fitted_model,
    frame: pd.DataFrame,
    treatment: int,
    adjustment_set: tuple[str, ...],
):
    ...


def summarize_results(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    ...
```

Các hàm không được ngầm lấy `df`, `model`, `seed` hoặc `config` từ global
scope. Đối tượng framework như `DiscreteBayesianNetwork` và `torch.nn.Module`
được phép sử dụng ở bên trong hàm factory/training; không biến chúng thành
state dùng chung giữa các seed. Dùng dataclass cho cấu hình và kết quả, không
dùng dictionary lồng nhau thay cho schema cố định.

---

## 1. B0 - Baseline CBN (Cốt lõi - Slide 57, 59)

1. Học cấu trúc DAG trên tập TRAIN bằng phương pháp truyền thống (`pgmpy` HillClimbSearch, PC, GES hoặc `bnlearn` hc/tabu) kết hợp danh sách cạnh cấm theo thời gian.
2. Tạo `DiscreteBayesianNetwork` với các cạnh hợp lệ:
   $C_i \rightarrow T$, $C_i \rightarrow Y$, $T \rightarrow Y$.
3. Khớp CPDs bằng API `pgmpy` hiện tại:
   `model.fit(frame, estimator=BayesianEstimator, prior_type="BDeu",
   equivalent_sample_size=5)`. Không khởi tạo estimator rồi gọi
   `get_parameters()`; method này không tồn tại trong `pgmpy 1.x`.
4. Gọi `check_model()`, nếu fail thì raise lỗi có ngữ cảnh.
5. Dùng `VariableElimination` để tính $P(Y \mid T)$ (quan sát) và xác suất can thiệp $q_{\text{B0}}(t, C_i) = P(Y=1 \mid T=t, C_i)$.
6. Chuẩn hóa nguy cơ trên tập kiểm thử $C_{\text{test}}$:
   $$\text{risk}(t) = \frac{1}{n_{\text{test}}} \sum_{i \in \text{TEST}} q(t, C_i), \quad \text{ATE} = \text{risk}(1) - \text{risk}(0)$$
   *(Tuyệt đối không dùng nhãn thực tế $Y_{\text{test}}$ để tính hiệu ứng can thiệp).*

---

## 2. H1 - Neural Parameters trên DAG B0 (Cốt lõi - Slide 60, 61)

- Cố định hoàn toàn cấu trúc DAG của B0 cho từng run.
- Giữ nguyên các biến số, trạng thái và phép chia tập Train/Val/Test.
- Thay thế CPD của $Y$ bằng mạng MLP nhận đầu vào **chỉ gồm các nút cha trực tiếp (parents) của $Y$** (không nhận hậu duệ hay các biến ngoài đồ thị).
- Huấn luyện mạng nơ-ron bằng Binary Cross-Entropy loss với Validation early stopping; lưu checkpoint tốt nhất.
- Liệt kê các tổ hợp giá trị rời rạc của các nút cha để sinh bảng CPD xác suất đã chuẩn hóa, sau đó gắn lại vào bản sao mạng Bayes của B0.
- Tính nguy cơ can thiệp trên tập kiểm thử theo cùng công thức chuẩn hóa Slide 61:
   $$q_{\text{H1}}(t, C_i) = P_{\text{H1}}(Y=1 \mid T=t, C_i)$$
   $$\text{risk}_{\text{H1}}(t) = \frac{1}{n_{\text{test}}} \sum_{i \in \text{TEST}} q_{\text{H1}}(t, C_i), \quad \text{ATE}_{\text{H1}} = \text{risk}(1) - \text{risk}(0)$$

---

## 3. H2 - Neural Structure Learning (Cốt lõi - Slide 62, 63)

- Tối ưu hóa gradient liên tục trên ma trận trọng số kề $W$ (DAG-GNN / nonlinear NOTEARS) trên cùng các biến và dữ liệu Train như B0.
- **Ma trận mặt nạ ràng buộc miền $M$ (Slide 63):**
  $$W_{\text{eff}} = M \odot W$$
  Đặt $M_{ij} = 0$ cho các cạnh vi phạm trật tự thời gian ($Y \rightarrow T$, $Y \rightarrow C_k$, $T \rightarrow C_k$) và đường chéo $M_{ii} = 0$.
- Kích hoạt ràng buộc phi chu trình $h(W_{\text{eff}}) = \text{Tr}(e^{W_{\text{eff}} \circ W_{\text{eff}}}) - d = 0$.
- Áp dụng ngưỡng lọc cạnh (thresholding) trên Validation set; áp dụng quy tắc sửa chu trình (cycle repair) nhất quán nếu xuất hiện chu trình.
- Chạy lại nhận diện Backdoor trên đồ thị H2 học được; khớp CPDs phi nơ-ron bằng cùng họ ước lượng làm mịn như B0 (`BayesianEstimator` BDeu).
- Đánh giá dự báo trên $Y_{\text{test}}$ và chuẩn hóa hiệu ứng can thiệp trên cùng tập tham chiếu $C_{\text{test}}$.
- Nếu không hội tụ hoặc sinh chu trình không sửa được, ghi nhận trạng thái `Failed` theo Slide 79, không tự động coi là DAG hợp lệ.

---

## 4. Kiểm định DoWhy Refutations (Cốt lõi - Slide 70, 71)

Bắt buộc tích hợp 2 bài kiểm tra bác bỏ cho từng mô hình:
1. `placebo_treatment_refuter`: Hoán vị ngẫu nhiên nhãn $T$; hiệu ứng kỳ vọng triệt tiêu về xấp xỉ 0 ($p > 0.05$).
2. `data_subset_refuter`: Ước lượng lại trên tập con 80% dữ liệu để đánh giá độ nhạy mẫu.

---

## 5. Các thành phần mở rộng tùy chọn (Optional Extensions - Slide 44, 51, 53, 64–67, 80)

*(Các mục này không bắt buộc cho điểm số Rubric cốt lõi, dùng làm bài tập mở rộng hoặc đưa vào Phụ lục bài báo)*:

### 5.1. Nghiên cứu bóc tách có kiểm soát (Ablations A1–A3)
Áp dụng đúng nguyên tắc "One change only" so với mô hình cha:

| Mã | Parent | Thay đổi duy nhất | Mục tiêu kiểm chứng |
|---|---|---|---|
| **A1** | H1 | Dropout rate = 0 (bỏ điều chuẩn) | Đo lường mức độ quá khớp của mạng nơ-ron |
| **A2** | H2 | Mở đúng một cạnh bị cấm trong mask $M$ | Kiểm tra độ nhạy của thuật toán học cấu trúc trước vi phạm miền |
| **A3** | B0 | Thay BDeu bằng `MaximumLikelihoodEstimator` | So sánh làm mịn Bayes vs MLE tần số khi có ô dữ liệu thưa |

Mỗi ablation dùng cùng seed, split, và cách tính hiệu ứng với model cha. Xuất $\Delta = \text{variant} - \text{parent}$.

### 5.2. Mô hình kết hợp H1+H2 (Slide 80)
Gắn các phân phối nơ-ron của H1 lên cấu trúc DAG học được từ H2 để phân tách đóng góp độc lập của cấu trúc và tham số.

### 5.3. Bài toán mô phỏng chân lý Synthetic Tutoring & DoWhy GCM (Slide 72, 73)
Bài toán minh họa lớp học 20,000 dòng có chân lý $ATE_{\text{true}} = 0.30$ ($P(Y=1\mid \text{do}(T=0))=0.30$, $P(Y=1\mid \text{do}(T=1))=0.60$) và kiểm chứng mô phỏng can thiệp DoWhy GCM.
