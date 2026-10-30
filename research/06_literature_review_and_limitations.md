# 06. TỔNG QUAN HỌC THUẬT, THẢO LUẬN & CÁC HẠN CHẾ CỦA BỘ DỮ LIỆU UCI ID 697
> **Chủ đề:** Khảo cứu tài liệu, đào sâu các thảo luận học thuật, góc nhìn từ Kaggle S4E6 và các hướng tiếp cận nhân quả mới.  
> **Bộ dữ liệu:** Predict Students' Dropout and Academic Success (UCI ID: 697 / Realinho et al., 2021)  
> **Trạng thái:** Bổ sung tư liệu cho Phần 2.2 (Related Work), 2.5 (Design) và 2.7 (Discussion) của Báo cáo.

---

## 1. Nguồn gốc & Bối cảnh Thảo luận Cộng đồng

Bộ dữ liệu do nhóm nghiên cứu của **TS. Valentim Realinho** (Viện Bách khoa Portalegre, Bồ Đào Nha) công bố trên tạp chí *MDPI Data (2021)* nhằm hỗ trợ giảm thiểu tỷ lệ bỏ học đại học.

Bộ dữ liệu này đã trở thành tâm điểm của cộng đồng phân tích dữ liệu toàn cầu thông qua:
1. Hơn 100+ bài báo trích dẫn trên các hội nghị hàng đầu về Educational Data Mining (EDM), Learning Analytics & Knowledge (LAK).
2. Cuộc thi **Kaggle Playground Series Season 4, Episode 6 (S4E6)** thu hút hơn 3,000+ đội thi và Grandmasters tranh tài về mô hình hóa dữ liệu học tập.

---

## 2. Các Hạn chế Cốt tử của Dữ liệu (Critical Limitations & Pitfalls)

Được tổng hợp từ các bài báo khoa học và các cuộc thảo luận chuyên sâu của cộng đồng nghiên cứu:

### 2.1. Tính Mập mờ của Nhãn "Enrolled" (Fuzzy Class Boundary & Censoring)
* **Vấn đề:** Dữ liệu gốc chia thành 3 lớp: `Graduate`, `Dropout` và `Enrolled`. Tuy nhiên, lớp `Enrolled` đại diện cho các sinh viên học kéo dài quá thời hạn chuẩn (chưa tốt nghiệp nhưng cũng chưa chính thức bỏ học).
* **Bản chất thống kê:** Trong lý thuyết Phân tích Sinh tồn (Survival Analysis), đây là hiện tượng **Dữ liệu bị cắt bên phải (Right-Censored Data)** — biến cố cuối cùng chưa xảy ra tại thời điểm chốt sổ dữ liệu.
* **Hậu quả:** Nếu đưa cả 3 lớp vào mô hình phân loại thông thường, ranh giới quyết định sẽ bị nhòe (fuzzy boundary), vì sinh viên `Enrolled` có thể đỗ hoặc trượt trong tương lai.
* **Giải pháp chuẩn:** Trong phân tích nhân quả, các nhà nghiên cứu lọc bỏ nhóm này hoặc quy ước rõ ràng về một biến kết quả nhị phân hoàn tất ($N = 3,630$).

### 2.2. Bẫy Rò rỉ Dữ liệu & Biến Trung gian (Mediator Trap / Temporal Data Leakage)
* **Vấn đề:** Các biến kết quả học tập học kỳ 2 (như `Curricular units 2nd sem (approved)`, `Curricular units 2nd sem (grade)`) có mức độ tương quan cực cao với việc tốt nghiệp (chiếm 60–70% Feature Importance trên Kaggle).
* **Góc nhìn Dự báo (Prediction) vs Nhân quả (Causality):**
  * *Trong Machine Learning thuần túy:* Sử dụng biến kỳ 2 cho điểm F1/Accuracy rất cao (~85–90%).
  * *Trong Can thiệp sớm (Early Intervention):* Đợi đến khi sinh viên thi trượt hết học kỳ 2 mới phát hiện nguy cơ thì đã quá muộn để can thiệp giữ chân họ!
  * *Trong Causal DAG:* Biến kỳ 2 diễn ra **SAU** khi sinh viên được cấp học bổng. Do đó, chúng là **Biến trung gian (Mediators)**. Đưa chúng vào tập điều chỉnh cửa sau sẽ chặn mất dòng nhân quả của học bổng và gây ra **Thiên lệch biến trung gian (Mediator Collider Bias)**.

### 2.3. Thiếu vắng các Biến Hành vi Động theo Thời gian thực (Static vs. Dynamic Telemetry)
* **Vấn đề:** Toàn bộ dữ liệu UCI ID 697 là dữ liệu dạng bảng tĩnh (static aggregated indicators) đo tại các cột mốc rời rạc (tuyển sinh, kết thúc kỳ học).
* **Thiếu sót:** Hoàn toàn vắng bóng các biến hành vi động trong Hệ thống quản lý học tập (LMS/VLE như Moodle, Canvas): số lượt click tài liệu, thời gian nộp bài tập, tần suất tham gia diễn đàn, tỷ lệ chuyên cần trên lớp.
* **Hậu quả:** Mô hình có thể chỉ ra *ai* đang gặp rủi ro vào cuối kỳ, nhưng không thể cung cấp tín hiệu cảnh báo sớm hàng tuần để cố vấn học tập hành động ngay lập tức.

### 2.4. Các Yếu tố Gây nhiễu Chưa Quan sát (Unmeasured Confounders)
* **Vấn đề:** Các động lực nội tâm quan trọng nhất chi phối thành công học tập hoàn toàn không được đo lường trong bộ dữ liệu:
  * Động lực học tập cá nhân (Intrinsic motivation / Grit).
  * Sức khỏe thể chất và tâm lý (Mental health / Depression / Stress).
  * Thời gian đi làm thêm kiếm sống ngoài giờ (Part-time / Full-time job hours).
* **Hậu quả:** Các biến vĩ mô như GDP, lạm phát được gán chung cho toàn vùng chứ không đo được áp lực tài chính thực tế lên từng cá nhân (Ecological Fallacy). Đây là hạn chế lớn nhất khi bảo vệ giả định *No Unmeasured Confounding*.

### 2.5. Giới hạn Khái quát hóa Địa lý & Thể chế (External Validity)
* Dữ liệu chỉ thu thập tại **duy nhất 1 viện bách khoa tại Bồ Đào Nha** theo chuẩn tín chỉ Châu Âu (ECTS).
* Không thể suy diễn nguyên xi các kết luận chính sách cho các trường đại học tại Bắc Mỹ hay các trường tại Việt Nam nếu không tái hiệu chỉnh miền dữ liệu (Domain Adaptation).

---

## 3. Chuyển Dịch Tư Duy: Từ "Dự báo (Prediction)" sang "Can thiệp Nhân quả (Prescription)"

Cộng đồng nghiên cứu Giáo dục (EDM/LAK) ngày càng phản đối cách tiếp cận Machine Learning cổ điển:

$$\text{Prediction: } P(Y \mid X) \quad \xrightarrow{\text{Cơ bản}} \quad \text{"Ai có nguy cơ bỏ học?"}$$
$$\text{Prescription: } P(Y \mid \text{do}(T)) \quad \xrightarrow{\text{Nhân quả}} \quad \text{"Cần can thiệp chính sách gì để cứu sinh viên đó?"}$$

* **Nghịch lý tương quan:** Trong dữ liệu thô, sinh viên có học bổng có tỷ lệ tốt nghiệp cao hơn hẳn. Nhưng điều này là do sinh viên giỏi sẵn mới xin được học bổng ($Z \rightarrow T$ và $Z \rightarrow Y$).
* **Đóng góp của Mạng Bayes Nhân quả (CBN):** Áp dụng toán tử phẫu thuật đồ thị $\text{do}(T=t)$ để cắt đứt liên kết sai lệch của biến nền, từ đó đo lường được **Hiệu ứng nhân quả thuần túy của đồng tiền học bổng**.

---

## 4. Những Hướng Tiếp cận Mới & Cơ hội Mở rộng Học thuật

### Hướng 1: Mạng Bayes Nhân quả Lai ghép Học sâu (Đề tài Hiện tại của Nhóm)
* **Điểm đột phá:** Khắc phục hạn chế của mạng Bayes cổ điển (khó mô hình hóa quan hệ phi tuyến phức tạp) bằng cách dùng MLP nơ-ron sâu cho cơ chế tham số (H1) và dùng tối ưu hóa gradient liên tục để học đồ thị (H2: DAG-GNN/NOTEARS có mask).
* **Ưu thế:** Vừa giữ được tính diễn giải và định danh nhân quả bằng do-calculus, vừa đạt độ khớp dữ liệu cao của Deep Learning.

### Hướng 2: Ước lượng Hiệu ứng Can thiệp Không đồng nhất (CATE / Uplift Modeling)
* **Ý tưởng:** Học bổng không tác động như nhau lên mọi sinh viên.
* **Mô hình mới:** Causal Forests, Meta-learners (T-Learner, X-Learner của thư viện `EconML` / `DoWhy`).
* **Giá trị thực tiễn:** Tìm ra nhóm sinh viên có hiệu ứng $\text{CATE} = \mathbb{E}[Y \mid \text{do}(T=1), Z] - \mathbb{E}[Y \mid \text{do}(T=0), Z]$ cao nhất (ví dụ: nhóm sinh viên nghèo có điểm đầu vào trung bình) để nhà trường cấp học bổng trúng đích, thay vì cấp cho sinh viên đã quá giỏi (nhóm sẽ tốt nghiệp dù có học bổng hay không).

### Hướng 3: Phân tích Sinh tồn Nhân quả (Causal Survival Analysis)
* **Ý tưởng:** Coi việc bỏ học là biến cố theo thời gian (Time-to-Event).
* **Mô hình mới:** Cox Proportional Hazards Model hoặc DeepSurv kết hợp can thiệp nhân quả.
* **Lợi ích:** Giải quyết triệt để lớp `Enrolled` như một trường hợp Right-Censoring thay vì phải loại bỏ khỏi tập dữ liệu.

### Hướng 4: Học máy Công bằng Nhân quả (Counterfactual Fairness)
* **Ý tưởng:** Đảm bảo quyết định cấp học bổng và đánh giá tốt nghiệp không mang thiên lệch phân biệt đối xử (Algorithmic Bias) đối với giới tính (`Gender`), hoàn cảnh gia đình hoặc độ tuổi (`Age at enrollment`).
* **Phương pháp:** Kiểm tra xem nếu thay đổi phản thực tế (counterfactual) giới tính của sinh viên thì quyết định mô hình có thay đổi không ($Y_{G \leftarrow g} \stackrel{?}{=} Y_{G \leftarrow g'}$).
