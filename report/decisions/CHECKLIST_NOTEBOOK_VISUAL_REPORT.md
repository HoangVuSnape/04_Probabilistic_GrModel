# BẢNG KIỂM TRA ĐỐI CHIẾU HÌNH ẢNH & BÁO CÁO KHOA HỌC (CHECKLIST_NOTEBOOK_VISUAL_REPORT.md)

> **Mục đích:** Xác minh tính toàn vẹn, nguồn gốc mã nguồn, thông số định lượng và tính nhất quán học thuật của toàn bộ tài sản hình ảnh và báo cáo tổng hợp [NOTEBOOK_VISUAL_ANALYSIS_REPORT.md](../NOTEBOOK_VISUAL_ANALYSIS_REPORT.md).

---

## 1. BẢNG KIỂM TRA HÌNH ẢNH EDA (`artifacts/eda/`)

| STT | Tên file hình ảnh | Tồn tại trên đĩa | Kích thước | Notebook sinh ảnh | Trạng thái đối chiếu số liệu |
|:---:|---|:---:|:---:|---|:---:|
| 1 | `uci_core_eda.png` | [x] | 128 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: Điểm đầu vào $95-190$, Học bổng $24.8\%$ |
| 2 | `kaggle_core_eda.png` | [x] | 138 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: $76.518$ dòng, CTGAN smoothing, Học bổng $24.8\%$ |
| 3 | `uci_kaggle_comparison.png` | [x] | 71 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: $KS=0.0573$, Wasserstein $=1.761$ |
| 4 | `uci_correlation_heatmap.png` | [x] | 334 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: $37 \times 37$ Pearson block ($r > 0.85$ cho HK1/HK2) |
| 5 | `kaggle_correlation_heatmap.png` | [x] | 334 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: Bảo toàn cụm tương quan học thuật và gia cảnh |
| 6 | `uci_mutual_information.png` | [x] | 70 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: Top 1 Tuition fees ($0.101$), Top 2 Scholarship ($0.067$) |
| 7 | `kaggle_mutual_information.png` | [x] | 74 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: Top 1 Tuition fees ($0.119$), Top 3 Scholarship ($0.105$) |
| 8 | `uci_advanced_quality.png` | [x] | 272 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: Course ID 9000-10000, $T=1 \to 75.8\%$ Tốt nghiệp |
| 9 | `kaggle_advanced_quality.png` | [x] | 275 KB | `eda_uci697_kaggle_s4e6.ipynb` | Khớp: $T=0 \to 36.2\%$ vs $T=1 \to 82.5\%$ Tốt nghiệp |

---

## 2. BẢNG KIỂM TRA HÌNH ẢNH XUẤT BẢN KHOA HỌC (`artifacts/figures/`)

| STT | Tên file hình ảnh | Định dạng kép (.png & .svg) | Kích thước | Notebook / Script gốc | Trạng thái đối chiếu số liệu |
|:---:|---|:---:|:---:|---|:---:|
| 1 | `fig1_cinelli_hazlett_contour.png` | [x] Có cả .svg | 402 KB | `causal_visualization_and_synthesis.ipynb` | Khớp: $\hat{\tau}=+0.2268$, $RV_{q=1}=19.1\%$ |
| 2 | `fig2_policy_learning_curve.png` | [x] Có cả .svg | 330 KB | `causal_visualization_and_synthesis.ipynb` | Khớp: Tại $26.7\%$ ngân sách, $+4.52\%$ Tốt nghiệp (+164 SV) |
| 3 | `fig3_factorial_matrix_tradeoff.png` | [x] Có cả .svg | 295 KB | `causal_visualization_and_synthesis.ipynb` | Khớp: Neural CPD cải thiện Brier ($0.1667$), ATE $\approx +0.23$ |
| 4 | `fig4_mediation_path_decomposition.png` | [x] Có cả .svg | 218 KB | `causal_visualization_and_synthesis.ipynb` | Khớp: $NDE=78.5\%$, $NIE=21.5\%$, $\alpha=+0.86$, $\beta=+5.41\%$ |
| 5 | `fig5_heterogeneous_cate_forest.png` | [x] Có cả .svg | 238 KB | `causal_visualization_and_synthesis.ipynb` | Khớp: SV lớn tuổi $CATE=+28.0\%$ vs đúng tuổi $+16.1\%$ |
| 6 | `fig6_uci_vs_kaggle_distributions.png` | [x] Có cả .svg | 379 KB | `eda_and_causal_benchmark_kaggle_s4e6.ipynb` | Khớp: 4 panels, độ lệch tỷ lệ $< 2.5\%$, $N_1=4.424, N_2=76.518$ |
| 7 | `fig7_cross_dataset_causal_generalization.png` | [x] Có cả .svg | 299 KB | `eda_and_causal_benchmark_kaggle_s4e6.ipynb` | Khớp: UCI $\to$ Kaggle $ATE=+0.2143$, AUROC $0.8537$, Brier $0.1539$ |

---

## 3. BẢNG KIỂM TRA TÍNH TOÀN VẸN CỦA BÁO CÁO ([NOTEBOOK_VISUAL_ANALYSIS_REPORT.md](../NOTEBOOK_VISUAL_ANALYSIS_REPORT.md))

- [x] Đã trích dẫn đầy đủ đường dẫn tuyệt đối dạng `file:///` cho tất cả các file hình ảnh và notebook.
- [x] Đã nhúng toàn bộ ảnh trực tiếp vào bài báo cáo bằng cú pháp Markdown chuẩn `![caption](file:///...)`.
- [x] Đã giải thích cặn kẽ ý nghĩa kinh tế - xã hội và cơ chế can thiệp giáo dục cho từng hình ảnh.
- [x] Đã kiểm tra tính nhất quán giữa các ước lượng ATE / CATE trong báo cáo với bảng dữ liệu thực nghiệm CSV tại `artifacts/tables/`.
- [x] Đã phân định rõ ranh giới thời gian (Temporal separation): 24 biến trước nhập học vs 12 biến sau nhập học, bảo đảm không vi phạm giả định can thiệp nhân quả do-calculus.
