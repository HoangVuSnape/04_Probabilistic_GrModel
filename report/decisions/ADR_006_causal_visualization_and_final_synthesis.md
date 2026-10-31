# ADR-006: HỆ THỐNG TRỰC QUAN HÓA NHÂN QUẢ HỌC THUẬT (300 DPI / VECTOR SVG) VÀ TỔNG HỢP KẾT LUẬN TOÀN DIỆN CHO ĐỒ ÁN MÔ HÌNH ĐỒ THỊ XÁC SUẤT
> **Trạng thái:** ACCEPTED (ĐÃ CHẤP THUẬN & HOÀN THÀNH TÍCH HỢP)  
> **Ngày phê duyệt:** 2026-10-31  
> **Người đề xuất:** Nhóm nghiên cứu Causal ML  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Tài liệu tham chiếu:** Slide 70–73, 80, 86 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).  
> **File Kiểm tra Nghiệm thu:** [CHECKLIST_ADR_006.md](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/decisions/CHECKLIST_ADR_006.md)  
> **Notebook Triển khai:** [causal_visualization_and_synthesis.ipynb](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_visualization_and_synthesis.ipynb)  
> **Báo cáo Tổng kết Học phần:** [FINAL_COURSE_PROJECT_REPORT.md](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/FINAL_COURSE_PROJECT_REPORT.md)

---

## 1. Bối cảnh & Mục tiêu (Context & Motivation)

Sau khi hoàn thành 4 giai đoạn thực nghiệm chuyên sâu:
1. **Kiểm toán dữ liệu & Sẵn sàng Nhân quả** (ADR-001, ADR-002, Notebook 1).
2. **Bộ ba Mô hình Cốt lõi B0, H1, H2 & Kiểm thử Phản bác DoWhy** (ADR-003, Notebook 2).
3. **Biến Kết quả Mở rộng, Phân rã Trung gian & Hiệu ứng Không đồng nhất CATE** (ADR-004, Notebook 3).
4. **Ma trận Yếu tố $2 \times 2$ (H3), Phân tích Độ nhạy Cinelli-Hazlett & Học Chính sách Can thiệp Tối ưu** (ADR-005, Notebook 4).

Đồ án cần một **cơ chế trực quan hóa khoa học đạt tiêu chuẩn công bố quốc tế** (chuẩn Nature / ICML: 300 DPI PNG và Scalable Vector Graphics SVG) và một **Báo cáo Khoa học Tổng kết Đồ án hoàn chỉnh** để tổng hợp toàn bộ các phát kiến thực nghiệm, phục vụ hội đồng chấm bảo vệ theo quy định Slide 86.

---

## 2. Các Quyết định Kiến trúc & Đặc tả Biểu đồ (Figures Specification)

Toàn bộ hệ thống trực quan hóa được sinh tự động bởi script [`tools/generate_causal_visualizations.py`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/tools/generate_causal_visualizations.py) và lưu trữ đồng thời ở hai định dạng:
* `.png` (độ phân giải 300 DPI, hiển thị tối ưu trong Notebook và Slide).
* `.svg` (đồ họa vector vô hạn độ phóng đại, phục vụ in ấn luận văn và tài liệu PDF).

```
                             [HỆ THỐNG TRỰC QUAN HÓA ADR-006]
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
      ▼                  ▼                   ▼                   ▼                  ▼
  [HÌNH 1]           [HÌNH 2]            [HÌNH 3]            [HÌNH 4]           [HÌNH 5]
Cinelli Contour     Policy Curves       2x2 Trade-off     Mediation Diagram   CATE Forest Plot
(Sensitivity RV)   (Budget Dividend)   (Brier vs AUROC)   (NDE vs NIE Paths)  (Subgroups 95% CI)
```

### 2.1. Hình 1: Contour Plot Độ nhạy Nhiễu ẩn Cinelli & Hazlett (`fig1_cinelli_hazlett_contour`)
* **Trục tung & Trục hoành:** $R^2_{Y \sim U \mid Z}$ và $R^2_{T \sim U \mid Z}$ chạy từ $0.00$ đến $0.35$.
* **Đường mức:** Các đường đồng mức $\hat{\tau}_{\text{adj}}$ giảm dần từ $+0.20$ về $0.00$ (đường nét đứt màu đỏ thể hiện biên triệt tiêu hiệu ứng).
* **Vùng tô đỏ bóng mờ:** Miền $\hat{\tau}_{\text{adj}} \le 0$ (vùng hiệu ứng bị vô hiệu hóa).
* **Điểm Giá trị Bền vững (Robustness Value):** Tọa độ chéo $(19.09\%, 19.09\%)$.
* **Điểm đối chuẩn thực nghiệm:** 5 biến gây nhiễu quan sát ($Z_1, Z_2, Z_3, Z_4, Z_5$) được chấm tọa độ hình thoi, chứng minh toàn bộ các yếu tố thực tế đều nằm sâu trong vùng an toàn.

### 2.2. Hình 2: Đường cong Hiệu quả Học Chính sách Tối ưu (`fig2_policy_learning_curve`)
* **Trục hoành:** Tỷ lệ phân bổ ngân sách ($0\%$ đến $100\%$, tương ứng $0$ đến $3,630$ suất học bổng).
* **Trục tung:** Tỷ lệ tốt nghiệp toàn trường dự kiến ($48\%$ đến $75\%$).
* **3 Đường quỹ đạo:**
  - *Chính sách Tối ưu Nhân quả (CATE Optimal):* Đường cong lồi dốc đứng lúc đầu thể hiện quy luật lợi suất biên giảm dần.
  - *Chính sách Ưu tiên Cần thiết (Need-based Debtor):* Đường đứt đoạn màu xanh dương.
  - *Chính sách Phân bổ Ngẫu nhiên (Random):* Đường thẳng tuyến tính màu cam.
* **Điểm thực tế lịch sử (Factual Point):** Tọa độ $(26.7\%, 60.85\%)$.
* **Dấu ngoặc Thặng dư Nhân quả (Causal Dividend Bracket):** Minh họa trực quan bước nhảy $+4.52\%$ tỷ lệ tốt nghiệp (tăng $+164$ sinh viên tốt nghiệp/năm) ở chi phí ngân sách bổ sung bằng $0$.

### 2.3. Hình 3: So sánh Đa chiều Ma trận Yếu tố 2x2 (`fig3_factorial_matrix_tradeoff`)
* **Subplot A (Predictive Calibration):** Scatter plot giữa Brier Score (trục X, càng thấp càng tốt) và AUROC (trục Y, càng cao càng tốt) với thanh sai số $\pm 1 \text{ SD}$ trên 5 seeds.
* **Subplot B (Causal Effect Robustness):** Horizontal Bar Chart biểu diễn ước lượng $ATE \pm 95\%$ Confidence Interval cho 4 mô hình $\{B_0, H_1, H_2, H_3\}$ so với đường chuẩn $+0.23$.

### 2.4. Hình 4: Sơ đồ Đường dẫn Phân rã Trung gian Nhân quả (`fig4_mediation_path_decomposition`)
* Sơ đồ hộp đồ thị nhân quả phân tầng thời gian $T \to M \to Y$.
* Hiệu ứng Trực tiếp Tự nhiên: $\text{NDE} = +16.96\%$ ($78.5\%$ tổng hiệu ứng).
* Hiệu ứng Gián tiếp Tự nhiên qua tín chỉ đỗ HK1: $\text{NIE} = +4.65\%$ ($21.5\%$ tổng hiệu ứng).
* Tổng hiệu ứng: $\text{TE} = +21.61\%$.

### 2.5. Hình 5: Biểu đồ Forest Plot Hiệu ứng Không đồng nhất (`fig5_heterogeneous_cate_forest`)
* Forest plot chuẩn y khoa với các điểm ước lượng tâm và khoảng tin cậy $95\%$ nằm ngang.
* Minh họa sự chênh lệch hiệu ứng giữa sinh viên lớn tuổi ($+28.02\%$) so với đúng tuổi ($+16.08\%$) và nhóm sinh viên nợ học phí ($+20.27\%$, tăng gấp đôi tỷ lệ tốt nghiệp từ $19.74\%$ lên $40.01\%$).

---

## 3. Danh mục Notebooks & Cấu trúc Hồ sơ Đồ án

Hệ thống đồ án chính thức phân định thành **5 Notebooks độc lập chuyên biệt**:
1. [`notebook/eda_uci697_kaggle_s4e6.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/eda_uci697_kaggle_s4e6.ipynb) — Giai đoạn 1: EDA, Kiểm toán tính sẵn sàng nhân quả, Phân tích Positivity & Lưu trữ Splits.
2. [`notebook/causal_modeling_uci697.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_modeling_uci697.ipynb) — Giai đoạn 2: Bộ ba mô hình cốt lõi $B_0, H_1, H_2$, Đánh giá ngoài mẫu Test Y, Can thiệp $do(T)$ và Kiểm thử phản bác DoWhy.
3. [`notebook/causal_advanced_outcomes_uci697.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_advanced_outcomes_uci697.ipynb) — Giai đoạn 3: Biến kết quả đa chiều (GPA HK2, Velocity, Full Cohort N=4,424), Causal Mediation & CATE.
4. [`notebook/causal_policy_and_sensitivity_uci697.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_policy_and_sensitivity_uci697.ipynb) — Giai đoạn 4: Ma trận yếu tố $2 \times 2$ ($H_3$), Độ nhạy Cinelli-Hazlett, Học chính sách tối ưu & Synthetic Benchmark.
5. [`notebook/causal_visualization_and_synthesis.ipynb`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/notebook/causal_visualization_and_synthesis.ipynb) — Giai đoạn 5: Trực quan hóa học thuật 300 DPI và Tổng hợp Kết luận Toàn diện.

---

## 4. Quyết định Chấp thuận & Nghiệm thu

* **Chấp thuận Báo cáo & Trực quan hóa:** Toàn bộ 5 biểu đồ và 5 notebook đã hoàn thành và vượt qua quy trình kiểm thử tự động.
* **Lưu hành Báo cáo Tổng kết:** Báo cáo khoa học chính thức được xuất bản tại [`report/FINAL_COURSE_PROJECT_REPORT.md`](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/report/FINAL_COURSE_PROJECT_REPORT.md).
