# BẢNG CHECKLIST KIỂM THỬ VÀ NGHIỆM THU ADR-006
> **Tài liệu kiểm định cho:** `report/decisions/ADR_006_causal_visualization_and_final_synthesis.md`  
> **Áp dụng cho:** Thư mục biểu đồ `artifacts/figures/`, Script sinh đồ thị `tools/generate_causal_visualizations.py`, File Notebook 5 `notebook/causal_visualization_and_synthesis.ipynb`, và Báo cáo Tổng kết `report/FINAL_COURSE_PROJECT_REPORT.md`  
> **Môn học:** Mô hình đồ thị có xác suất (IT701100) — HK1 2026–2027  
> **Quy chuẩn:** Slide 70–73, 80, 86 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf`).

---

## I. THÔNG TIN QUY CHUẨN & PHẠM VI NGHIỆM THU

- [x] **Mã quyết định kiến trúc:** ADR-006.
- [x] **Tên quyết định:** Hệ thống Trực quan hóa Nhân quả Học thuật (300 DPI / Vector SVG) và Tổng hợp Kết luận Toàn diện cho Đồ án Mô hình Đồ thị Xác suất.
- [x] **Số lượng biểu đồ học thuật:** 5 biểu đồ, xuất đồng thời cả định dạng PNG (300 DPI) và SVG.
- [x] **File notebook thực thi chính thức:** `notebook/causal_visualization_and_synthesis.ipynb` (độc lập, không ghi đè, chạy thành công 100%).
- [x] **Báo cáo tổng kết đồ án:** `report/FINAL_COURSE_PROJECT_REPORT.md`.

---

## II. CHECKLIST CHI TIẾT 5 BIỂU ĐỒ HỌC THUẬT

### 1. Hình 1: Contour Plot Độ nhạy Nhiễu ẩn Cinelli & Hazlett
- [x] **File lưu trữ:** `artifacts/figures/fig1_cinelli_hazlett_contour.png` (392.8 KB) & `fig1_cinelli_hazlett_contour.svg` (115.1 KB).
- [x] **Độ phân giải:** Đạt chuẩn 300 DPI.
- [x] **Đường mức:** Thể hiện đầy đủ các mức $\hat{\tau}_{\text{adj}} \in [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20]$.
- [x] **Đường ranh giới triệt tiêu hiệu ứng:** Nét đứt màu đỏ tại mức $0.00$.
- [x] **Vùng bóng mờ:** Thể hiện miền bị vô hiệu hóa ($\hat{\tau}_{\text{adj}} \le 0$).
- [x] **Tọa độ Robustness Value ($RV$):** Đánh dấu chính xác điểm $(19.09\%, 19.09\%)$.
- [x] **Tọa độ 5 Confounders quan sát:** Thể hiện đầy đủ $Z_1$ (Tuổi), $Z_2$ (Tuyển sinh), $Z_3$ (Mẹ), $Z_4$ (Nợ học phí), $Z_5$ (Học phí).

### 2. Hình 2: Đường cong Hiệu quả Học Chính sách Tối ưu
- [x] **File lưu trữ:** `artifacts/figures/fig2_policy_learning_curve.png` (323.0 KB) & `fig2_policy_learning_curve.svg` (137.8 KB).
- [x] **3 Quỹ đạo chính sách:** Đầy đủ CATE Optimal (đường cong xanh lá), Need-based (xanh dương đứt đoạn), Random (cam chấm).
- [x] **Điểm thực tế lịch sử (Factual):** Đánh dấu hình ngôi sao tại tọa độ $(26.7\%, 60.85\%)$.
- [x] **Khung ngoặc Thặng dư Nhân quả:** Minh họa rõ ràng khoảng tăng $+4.52\%$ tỷ lệ tốt nghiệp (+164 sinh viên) với chi phí ngân sách bổ sung bằng $0$.

### 3. Hình 3: So sánh Ma trận Yếu tố 2x2
- [x] **File lưu trữ:** `artifacts/figures/fig3_factorial_matrix_tradeoff.png` (288.7 KB) & `fig3_factorial_matrix_tradeoff.svg` (153.4 KB).
- [x] **Panel A (Predictive Trade-off):** Thể hiện tương quan giữa Brier Score và AUROC cho cả 4 mô hình ($B_0, H_1, H_2, H_3$) kèm thanh sai số $\pm 1 \text{ SD}$.
- [x] **Panel B (Causal Effect):** Biểu đồ thanh ngang ATE kèm khoảng tin cậy $95\%$, đối chiếu với đường chuẩn $+0.23$.

### 4. Hình 4: Sơ đồ Đường dẫn Phân rã Trung gian Nhân quả
- [x] **File lưu trữ:** `artifacts/figures/fig4_mediation_path_decomposition.png` (213.6 KB) & `fig4_mediation_path_decomposition.svg` (90.5 KB).
- [x] **Cấu trúc 3 hộp:** Thể hiện Treatment $T$, Mediator $M$ (Tín chỉ đỗ HK1), Outcome $Y$.
- [x] **Số liệu phân rã:** NDE $= +16.96\%$ ($78.5\%$), NIE $= +4.65\%$ ($21.5\%$), Total Effect $= +21.61\%$.

### 5. Hình 5: Biểu đồ Forest Plot Hiệu ứng Không đồng nhất
- [x] **File lưu trữ:** `artifacts/figures/fig5_heterogeneous_cate_forest.png` (232.7 KB) & `fig5_heterogeneous_cate_forest.svg` (130.3 KB).
- [x] **Các phân tầng:** Thể hiện đủ 7 phân tầng (Toàn thể, Độ tuổi $>20$ vs $\le 20$, Nợ học phí có vs không, Ở cùng gia đình vs Trọ).
- [x] **Thanh khoảng tin cậy:** Đầy đủ điểm tâm và thanh $95\%$ CI cho từng phân tầng.

---

## III. CHECKLIST KIỂM THỬ NOTEBOOK 5 & BÁO CÁO

- [x] **Kiểm thử tự động Notebook 5:** Chạy qua `tools/test_notebook_execution.py`, vượt qua 100% không phát sinh lỗi (`SUCCESS`, Exit code: 0).
- [x] **Cơ chế Fallback Display:** Tích hợp fallback class `Image` và hàm `display` đảm bảo notebook chạy trơn tru cả trong Jupyter IDE và Python CLI headless.
- [x] **Toàn vẹn Báo cáo Tổng kết:** File `report/FINAL_COURSE_PROJECT_REPORT.md` tổng hợp toàn bộ 5 chặng nghiên cứu, tích hợp các bảng biểu A, B, C và dẫn chiếu đầy đủ các file minh chứng.

---

## IV. KẾT LUẬN NGHIỆM THU

Hệ thống trực quan hóa học thuật và hồ sơ nghiệm thu **ADR-006** đạt tiêu chuẩn cao nhất về tính khoa học, thẩm mỹ học thuật và độ chính xác toán học, hoàn tất toàn bộ chu trình thực nghiệm của đồ án.
