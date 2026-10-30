# ADR-002: THIẾT KẾ BÀI TOÁN NHÂN QUẢ, ĐỒ THỊ DAG & CHỐNG RÒ RỈ DỮ LIỆU
> **Trạng thái:** ACCEPTED (ĐÃ CHẤP THUẬN)  
> **Ngày phê duyệt:** 2026-10-29  
> **Người đề xuất:** Nhóm nghiên cứu  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  

---

## 1. Bối cảnh (Context)
* Trong suy diễn nhân quả (Causal Inference), việc xác định sai biến can thiệp, sai trật tự thời gian hoặc đưa các biến hậu can thiệp (Post-treatment variables / Mediators) vào tập điều chỉnh cửa sau (Backdoor adjustment set) sẽ dẫn đến sai lệch nghiêm trọng (Mediator bias / Collider stratification bias).
* Giảng viên đã cảnh báo rõ tại Slide 40: *"G1 and G2 are earlier-period grades. Include them only if measured before the chosen intervention... Remove G3 from predictors after creating pass"*.
* Nhóm cần đưa ra quyết định kiến trúc chuẩn xác về việc chọn biến, thiết lập đồ thị Domain DAG và cô lập biến rò rỉ trên tập dữ liệu UCI ID 697.

---

## 2. Quyết định (Decisions)

### 2.1. Lựa chọn Biến can thiệp ($T$) và Biến kết quả ($Y$)
* **Biến can thiệp ($T$):** Chọn thuộc tính `Scholarship holder` ($0 = \text{Không có học bổng}, 1 = \text{Có học bổng}$).
  * *Lý do:* Đây là một quyết định chính sách tài chính có thể can thiệp được trong thực tế đời sống (actionable intervention policy).
* **Biến kết quả ($Y$):** Chọn thuộc tính `Target`, ánh xạ về nhị phân:
  $$Y = \begin{cases} 1 & \text{nếu Target} = \text{'Graduate'} \\ 0 & \text{nếu Target} = \text{'Dropout'} \end{cases}$$

### 2.2. Lựa chọn Tập biến nền ($Z$) & Chống rò rỉ dữ liệu (No Data Leakage)
* Tập biến gây nhiễu hợp lệ $\mathbf{Z}$ bắt buộc phải diễn ra **TRƯỚC** khi học bổng được cấp ($t_0 \prec t_1$):
  1. $Z_1$: `Admission grade` (Điểm xét tuyển đầu vào).
  2. $Z_2$: `Debtor` (Tình trạng nợ học phí lúc nhập học).
  3. $Z_3$: `Tuition fees up to date` (Đóng học phí đúng hạn lúc đầu khóa).
  4. $Z_4$: `Mother's qualification` (Trình độ học vấn của mẹ).
  5. $Z_5$: `Age at enrollment` (Độ tuổi khi nhập học).
* **Quy tắc loại trừ sống còn:** Toàn bộ các biến kết quả học tập của học kỳ 2 (như `Curricular units 2nd sem (grade)`, `Curricular units 2nd sem (approved)`) tuyệt đối **KHÔNG ĐƯỢC ĐƯA VÀO** tập điều chỉnh $\mathbf{Z}$. Chúng diễn ra sau khi học bổng đã phát huy tác dụng trong suốt năm học, do đó là các **Biến trung gian (Mediators)**. Đưa chúng vào sẽ làm triệt tiêu hiệu ứng nhân quả trực tiếp của học bổng và gây thiên lệch ước lượng.

### 2.3. Cấu trúc Đồ thị Nhân quả Cơ sở (B0 DAG)
* Cạnh nguyên nhân từ biến nền: $Z_i \rightarrow T$ và $Z_i \rightarrow Y, \forall i \in \{1, \dots, 5\}$.
* Cạnh hiệu ứng can thiệp: $T \rightarrow Y$.
* Không có chu trình (Acyclic) và trật tự thời gian được bảo vệ tuyệt đối.

### 2.4. Quy chuẩn Thực nghiệm Lặp lại (Multiple Seeds)
* Bắt buộc thực hiện đối sánh trên **5 hạt giống ngẫu nhiên (seeds $\ge 5$)**:
  $$\text{Seeds} = [42, \ 123, \ 456, \ 789, \ 999]$$
* Báo cáo đầy đủ Kỳ vọng, Rủi ro, và Khoảng tin cậy 95% để phản ánh độ bất định thống kê.

---

## 3. Hệ quả (Consequences)
* Cung cấp cơ sở lý luận vững chắc, chặt chẽ để bảo vệ đề tài trước Giảng viên trong buổi duyệt đề tài Midterm.
* Đảm bảo tính toán tử can thiệp $P(Y \mid \text{do}(T=t))$ phản ánh đúng bản chất nhân quả chứ không phải tương quan quan sát $P(Y \mid T=t)$.
