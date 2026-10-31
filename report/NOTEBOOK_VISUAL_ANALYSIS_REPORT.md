# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU HỆ THỐNG BIỂU ĐỒ & HÌNH ẢNH THỰC NGHIỆM (COMPREHENSIVE VISUAL & EMPIRICAL FIGURE REPORT)

> **Môn học:** Mô hình đồ thị có xác suất (Probabilistic Graphical Models - IT701100)  
> **Đề tài:** Mô hình Đồ thị Nhân quả Xác suất (Causal Bayesian Networks) Phân tích Hiệu ứng Can thiệp Học bổng lên Tỷ lệ Tốt nghiệp Đại học  
> **Dữ liệu thực nghiệm:** UCI Repository ID 697 ($N=4.424$) và Kaggle Playground S4E6 ($N=76.518$)  
> **Tài nguyên hình ảnh:** Toàn bộ biểu đồ EDA (`artifacts/eda/`) và 7 biểu đồ xuất bản khoa học (`artifacts/figures/fig1` - `fig7`)  

---

## MỤC LỤC

1. [Tổng quan Hệ thống Tài sản Hình ảnh Thực nghiệm](#1-tổng-quan-hệ-thống-tài-sản-hình-ảnh-thực-nghiệm)
2. [Notebook 1: Phân tích Khám phá Dữ liệu Kép (Dual-Dataset EDA)](#2-notebook-1-phân-tích-khám-phá-dữ-liệu-kép-dual-dataset-eda)
   - [2.1 Phân bố Đặc trưng Cốt lõi & Phân tầng Điều trị (`uci_core_eda.png` & `kaggle_core_eda.png`)](#21-phân-bố-đặc-trưng-cốt-lõi--phân-tầng-điều-trị)
   - [2.2 So sánh Trực tiếp Phân bố UCI vs Kaggle (`uci_kaggle_comparison.png`)](#22-so-sánh-trực-tiếp-phân-bố-uci-vs-kaggle)
   - [2.3 Cấu trúc Ma trận Tương quan Toàn diện (`uci_correlation_heatmap.png` & `kaggle_correlation_heatmap.png`)](#23-cấu-trúc-ma-trận-tương-quan-toàn-diện)
   - [2.4 Xếp hạng Thông tin Tương hỗ Mutual Information (`*_mutual_information.png`)](#24-xếp-hạng-thông-tin-tương-hỗ-mutual-information)
   - [2.5 Kiểm toán Chất lượng, Dị biệt & Thời gian (`*_advanced_quality.png`)](#25-kiểm-toán-chất-lượng-dị-biệt--thời-gian)
3. [Notebook 2 & 3: Khám phá Cấu trúc Đồ thị & Phân rã Cơ chế Trung gian](#3-notebook-2--3-khám-phá-cấu-trúc-đồ-thị--phân-rã-cơ-chế-trung-gian)
   - [3.1 Cơ chế Tác động Trực tiếp & Gián tiếp (`fig4_mediation_path_decomposition.png`)](#31-cơ-chế-tác-động-trực-tiếp--gián-tiếp)
4. [Notebook 4: Hiệu ứng Can thiệp Không đồng nhất trên Quần thể](#4-notebook-4-hiệu-ứng-can-thiệp-không-đồng-nhất-trên-quần-thể)
   - [4.1 Forest Plot Phân tích CATE theo Phân nhóm Sinh viên (`fig5_heterogeneous_cate_forest.png`)](#41-forest-plot-phân-tích-cate-theo-phân-nhóm-sinh-viên)
5. [Notebook 5: Đánh giá Độ nhạy Thiên lệch Biến ẩn & Tối ưu hóa Chính sách Can thiệp](#5-notebook-5-đánh-giá-độ-nhạy-thiên-lệch-biến-ẩn--tối-ưu-hóa-chính-sách-can-thiệp)
   - [5.1 Đồ thị Đường đồng mức Độ nhạy Cinelli-Hazlett (`fig1_cinelli_hazlett_contour.png`)](#51-đồ-thị-đường-đồng-mức-độ-nhạy-cinelli-hazlett)
   - [5.2 Đường cong Học Chính sách Can thiệp dưới Ràng buộc Ngân sách (`fig2_policy_learning_curve.png`)](#52-đường-cong-học-chính-sách-can-thiệp-dưới-ràng-buộc-ngân-sách)
6. [Notebook Synthesis: Đánh giá Ma trận Đa nhân tố Dự báo & Nhân quả](#6-notebook-synthesis-đánh-giá-ma-trận-đa-nhân-tố-dự-báo--nhân-quả)
   - [6.1 Đánh đổi Brier - AUROC vs Tính Nhất quán ATE (`fig3_factorial_matrix_tradeoff.png`)](#61-đánh-đổi-brier---auroc-vs-tính-nhất-quán-ate)
7. [Notebook 6: Thực nghiệm Quy mô Lớn 76.518 Dòng & Tính Tổng quát hóa Ngoại suy](#7-notebook-6-thực-nghiệm-quy-mô-lớn-76518-dòng--tính-tổng-quát-hóa-ngoại-suy)
   - [7.1 Kiểm định Mật độ Phân bố Dữ liệu Lớn (`fig6_uci_vs_kaggle_distributions.png`)](#71-kiểm-định-mật-độ-phân-bố-dữ-liệu-lớn)
   - [7.2 Bền vững Nhân quả qua Chuyển dịch Phân phối & Đánh giá OOD (`fig7_cross_dataset_causal_generalization.png`)](#72-bền-vững-nhân-quả-qua-chuyển-dịch-phân-phối--đánh-giá-ood)
8. [Tổng kết & Khuyến nghị Chính sách](#8-tổng-kết--khuyến-nghị-chính-sách)

---

## 1. TỔNG QUAN HỆ THỐNG TÀI SẢN HÌNH ẢNH THỰC NGHIỆM

Hệ thống thực nghiệm của đề tài sinh ra hai nhóm tài sản hình ảnh chính:
1. **Nhóm EDA Artifacts (`artifacts/eda/`):** 9 hình ảnh phân giải cao phân tích dữ liệu khám phá song song giữa bộ dữ liệu thực tế UCI 697 ($N=4.424$) và tập dữ liệu tổng hợp quy mô lớn Kaggle S4E6 ($N=76.518$).
2. **Nhóm Publication Figures (`artifacts/figures/`):** 7 hình ảnh đạt chuẩn xuất bản quốc tế (vector `.svg` và raster `.png` 300 DPI) trả lời trực tiếp các câu hỏi nghiên cứu nền tảng về nhân quả, độ nhạy, học chính sách và tính tổng quát hóa ngoài phân phối (OOD).

| STT | Tên file hình ảnh | Kích thước | Notebook xuất xứ | Nội dung khoa học cốt lõi |
|---|---|---|---|---|
| 1 | `uci_core_eda.png` / `kaggle_core_eda.png` | 128 KB / 138 KB | Notebook 1 | Phân bố điểm tuyển sinh, tỷ lệ học bổng và điểm theo độ tuổi |
| 2 | `uci_kaggle_comparison.png` | 71 KB | Notebook 1 | So sánh mật độ điểm đầu vào và tỷ lệ nhãn kết quả |
| 3 | `uci_correlation_heatmap.png` / `kaggle_correlation_heatmap.png` | 334 KB / 334 KB | Notebook 1 | Cấu trúc ma trận tương quan 37 biến |
| 4 | `uci_mutual_information.png` / `kaggle_mutual_information.png` | 70 KB / 74 KB | Notebook 1 | Xếp hạng thông tin tương hỗ của Top 15 đặc trưng trước nhập học |
| 5 | `uci_advanced_quality.png` / `kaggle_advanced_quality.png` | 272 KB / 275 KB | Notebook 1 | Phân tầng tỷ lệ tốt nghiệp theo can thiệp và kiểm toán thời gian |
| 6 | `fig1_cinelli_hazlett_contour.png` | 402 KB | Notebook 5 | Đường đồng mức độ nhạy thiên lệch biến ẩn $RV_{q=1} = 19.1\%$ |
| 7 | `fig2_policy_learning_curve.png` | 330 KB | Notebook 5 | Đường cong tối ưu hóa chính sách cấp học bổng theo ngân sách |
| 8 | `fig3_factorial_matrix_tradeoff.png` | 295 KB | Notebook Synthesis | Đánh đổi Brier-AUROC vs tính nhất quán ATE giữa 4 cấu hình |
| 9 | `fig4_mediation_path_decomposition.png` | 218 KB | Notebook 3 | Phân rã hiệu ứng trực tiếp NDE ($78.5\%$) và gián tiếp NIE ($21.5\%$) |
| 10 | `fig5_heterogeneous_cate_forest.png` | 238 KB | Notebook 4 | Forest plot CATE: Hiệu ứng vượt trội ở sinh viên lớn tuổi ($+28.0\%$) |
| 11 | `fig6_uci_vs_kaggle_distributions.png` | 379 KB | Notebook 6 | Kiểm định KS-test & Wasserstein giữa UCI 4.4k và Kaggle 76k |
| 12 | `fig7_cross_dataset_causal_generalization.png` | 299 KB | Notebook 6 | Bất biến nhân quả CBN qua chuyển dịch phân phối OOD Transfer |

---

## 2. NOTEBOOK 1: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU KÉP (DUAL-DATASET EDA)

### 2.1 Phân bố Đặc trưng Cốt lõi & Phân tầng Điều trị
*Đường dẫn tệp:* [uci_core_eda.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_core_eda.png) và [kaggle_core_eda.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_core_eda.png)

![UCI Core EDA](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_core_eda.png)
![Kaggle Core EDA](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_core_eda.png)

#### Phân tích chi tiết:
1. **Phân bố Điểm tuyển sinh đầu vào (Admission Grade - Panel trái):**
   - Trên tập UCI ($N=4.424$), điểm tuyển sinh dao động từ $95.0$ đến $190.0$ điểm trên thang 200, tập trung đối xứng đơn đỉnh quanh khoảng $120.0 - 130.0$ điểm (trung vị $126.6$, độ lệch chuẩn $13.2$).
   - Trên tập Kaggle ($N=76.518$), phân bố giữ nguyên kỳ vọng và phương sai nhưng có dạng trơn lượn sóng liên tục (smooth continuous curve) với các đỉnh phụ nhẹ tại các mốc điểm chẵn. Đây là đặc trưng vật lý điển hình sinh ra từ mạng đối nghịch tạo sinh có điều kiện (CTGAN), làm tăng tính đa dạng mà không làm lệch kỳ vọng thống kê.
2. **Tỷ lệ Phân bổ Học bổng (Scholarship Holder Counts - Panel giữa):**
   - Trên tập UCI: $3.325$ sinh viên không có học bổng ($75.2\%$) và $1.099$ sinh viên có học bổng ($24.8\%$).
   - Trên tập Kaggle: $57.533$ sinh viên không có học bổng ($75.2\%$) và $18.985$ sinh viên có học bổng ($24.8\%$).
   - Tỷ lệ phân bổ điều trị tự nhiên giữa 2 tập dữ liệu trùng khớp đến mức tuyệt đối ($24.8\%$), bảo đảm tính tương đồng hoàn hảo của cơ chế gán điều trị quan sát.
3. **Phân tầng Điểm tuyển sinh theo Độ tuổi và Kết quả Tốt nghiệp (Panel phải):**
   - Biểu đồ hộp (Boxplot) chia thành 3 phân nhóm tuổi: Đúng tuổi ($\le 20$ tuổi), Chuyển tiếp ($21 - 25$ tuổi), và Lớn tuổi ($> 25$ tuổi).
   - Ở cả 3 nhóm tuổi, nhóm sinh viên tốt nghiệp (màu cam) luôn có phân vị điểm tuyển sinh cao hơn rõ rệt so với nhóm bỏ học (màu xanh). Khoảng chênh lệch trung vị đạt $+5.2$ điểm ở nhóm $\le 20$ tuổi và $+6.1$ điểm ở nhóm $> 25$ tuổi.
   - Nhóm sinh viên $>25$ tuổi có phương sai điểm đầu vào rộng hơn nhiều, phản ánh sự đa dạng về nền tảng giáo dục trước đó (Previous qualification).

---

### 2.2 So sánh Trực tiếp Phân bố UCI vs Kaggle
*Đường dẫn tệp:* [uci_kaggle_comparison.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_kaggle_comparison.png)

![UCI vs Kaggle Comparison](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_kaggle_comparison.png)

#### Phân tích chi tiết:
1. **Mật độ Điểm tuyển sinh (Admission Grade Density Overlay - Panel trái):**
   - Đường cong mật độ của hai bộ dữ liệu lồng ghép khít khao trên toàn bộ dải giá trị từ 95 đến 180 điểm.
   - Khoảng cách thống kê Wasserstein chỉ đạt $1.761$ đơn vị điểm, và giá trị thống kê kiểm định Kolmogorov-Smirnov $KS = 0.0573$ ($p < 10^{-5}$ do kích thước mẫu cực lớn). Điều này bảo đảm không có hiện tượng co cụm hay cắt cụt biên (support truncation) trong tập dữ liệu Kaggle.
2. **Phân bố Nhãn Mục tiêu (Target Distribution - Panel phải):**
   - **Tốt nghiệp (Graduate):** UCI đạt $49.9\%$, Kaggle đạt $47.4\%$ (chênh lệch $-2.5\%$).
   - **Bỏ học (Dropout):** UCI đạt $32.1\%$, Kaggle đạt $33.1\%$ (chênh lệch $+1.0\%$).
   - **Đang theo học (Enrolled):** UCI đạt $18.0\%$, Kaggle đạt $19.5\%$ (chênh lệch $+1.5\%$).
   - Kết luận: Bộ dữ liệu Kaggle S4E6 mô phỏng trung thực cơ cấu nhãn thực tế của giáo dục đại học Bồ Đào Nha, với tỷ lệ tốt nghiệp áp đảo nhưng tỷ lệ bỏ học vẫn chiếm gần một phần ba tổng số sinh viên.

---

### 2.3 Cấu trúc Ma trận Tương quan Toàn diện
*Đường dẫn tệp:* [uci_correlation_heatmap.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_correlation_heatmap.png) và [kaggle_correlation_heatmap.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_correlation_heatmap.png)

![UCI Heatmap](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_correlation_heatmap.png)

#### Phân tích chi tiết:
1. **Cụm Tương quan Đẳng hướng Học thuật (Academic Performance Block):**
   - Cụm 12 biến đo lường kết quả học tập kỳ 1 và kỳ 2 (tín chỉ miễn giảm, tín chỉ đăng ký, số lần đánh giá, số tín chỉ đỗ, và điểm trung bình) tạo thành một khối đỏ đậm ở góc dưới bên phải ma trận với hệ số tương quan tuyến tính Pearson cực kỳ cao:
     - Số tín chỉ đỗ HK1 và HK2: $r = 0.923$.
     - Số tín chỉ đăng ký HK1 và HK2: $r = 0.956$.
     - Điểm số HK1 và HK2: $r = 0.889$.
   - **Hàm ý nhân quả:** Các biến này phản ánh cùng một năng lực tiềm ẩn (latent academic capability) và diễn ra **sau thời điểm nhập học**. Nếu đưa đồng thời các biến này vào làm biến kiểm soát nhập học sẽ tạo ra sai số Collider/M-bias nghiêm trọng.
2. **Cụm Nền tảng Gia đình (Socioeconomic Background Block):**
   - Nghề nghiệp của mẹ và bố có tương quan $r = 0.895$, trình độ học vấn của mẹ và bố có $r = 0.567$.
   - Quốc tịch và sinh viên quốc tế có tương quan $r = 0.817$.
3. **Mối liên hệ giữa Can thiệp Học bổng ($T$) và Tài chính ($Z_4, Z_5$):**
   - Học bổng có tương quan âm với Nợ học phí (`Debtor` - $r = -0.068$) và tương quan dương với Đóng học phí đúng hạn (`Tuition fees up to date` - $r = +0.137$), xác nhận vai trò bảo trợ tài chính trực tiếp của học bổng đối với sinh viên.

---

### 2.4 Xếp hạng Thông tin Tương hỗ Mutual Information
*Đường dẫn tệp:* [uci_mutual_information.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_mutual_information.png) và [kaggle_mutual_information.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_mutual_information.png)

![UCI MI](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_mutual_information.png)
![Kaggle MI](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_mutual_information.png)

#### Phân tích chi tiết:
1. **Đặc trưng Tài chính dẫn đầu Tuyệt đối:**
   - Trên cả UCI và Kaggle, biến `Tuition fees up to date` (Đóng học phí đúng hạn) đạt điểm Mutual Information cao nhất toàn diện ($MI = 0.101$ trên UCI và $MI = 0.119$ trên Kaggle). Điều này chứng minh khủng hoảng thanh toán tài chính là yếu tố dự báo sớm hàng đầu của hành vi bỏ học.
2. **Sự tương đồng thứ hạng giữa 2 tập dữ liệu:**
   - Trong Top 6 đặc trưng trước nhập học mạnh nhất:
     - `Scholarship holder` (Học bổng): Đứng thứ 2 trên UCI ($MI = 0.067$) và đứng thứ 3 trên Kaggle ($MI = 0.105$).
     - `Age at enrollment` (Độ tuổi nhập học): Đứng thứ 3 trên UCI ($MI = 0.061$) và đứng thứ 4 trên Kaggle ($MI = 0.104$).
     - `Course` (Chuyên ngành): Đứng thứ 5 trên UCI ($MI = 0.052$) và đứng thứ 2 trên Kaggle ($MI = 0.105$).
     - `Admission grade` (Điểm đầu vào): Đứng thứ 7 trên UCI ($MI = 0.033$) và đứng thứ 5 trên Kaggle ($MI = 0.103$).
   - Kết luận khoa học: Mối quan hệ thông tin phi tuyến giữa các biến quan sát và kết quả tốt nghiệp được bảo toàn nguyên vẹn trên cả tập thực tế và tập dữ liệu lớn.

---

### 2.5 Kiểm toán Chất lượng, Dị biệt & Thời gian
*Đường dẫn tệp:* [uci_advanced_quality.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_advanced_quality.png) và [kaggle_advanced_quality.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_advanced_quality.png)

![UCI Advanced Quality](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/uci_advanced_quality.png)
![Kaggle Advanced Quality](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/eda/kaggle_advanced_quality.png)

#### Phân tích chi tiết:
1. **Kiểm tra Outlier & Mã hóa Thực thể (Panel trên phải):**
   - Biểu đồ phân tán giá trị ngoại lai cho thấy các giá trị $> 8000$ chỉ xuất hiện duy nhất ở cột `Course`. Đây không phải lỗi nhập liệu mà là mã định danh chính thức của các chuyên ngành đào tạo thuộc hệ thống giáo dục đại học Bồ Đào Nha (ví dụ mã 9070 là Điều dưỡng, mã 9119 là Kỹ thuật máy tính).
2. **Phân tầng Tỷ lệ Tốt nghiệp theo Điều trị Can thiệp (Panel dưới trái):**
   - **Trên tập UCI:**
     - Khi Không có học bổng ($T=0$): Tỷ lệ Bỏ học là $38.5\%$, Tốt nghiệp là $41.3\%$.
     - Khi Có học bổng ($T=1$): Tỷ lệ Bỏ học giảm sâu xuống **$12.3\%$**, Tốt nghiệp tăng vọt lên **$75.8\%$** (chênh lệch thô $+34.5\%$).
   - **Trên tập Kaggle:**
     - Khi Không có học bổng ($T=0$): Tỷ lệ Bỏ học là $41.3\%$, Tốt nghiệp là $36.2\%$.
     - Khi Có học bổng ($T=1$): Tỷ lệ Bỏ học giảm còn **$7.1\%$**, Tốt nghiệp tăng vọt lên **$82.5\%$** (chênh lệch thô $+46.3\%$).
3. **Kiểm toán Thứ bậc Thời gian (Temporal Feature Audit - Panel dưới phải):**
   - Xác định chính xác **24 đặc trưng Baseline** (xảy ra trước hoặc tại thời điểm nhập học: tuổi, điểm, gia cảnh, tài chính, ngành học).
   - Xác định **12 đặc trưng Post-enrollment** (xảy ra trong quá trình học: kết quả học tập HK1/HK2).
   - Thiết lập nguyên tắc bất khả xâm phạm: **Không đưa 12 đặc trưng sau nhập học vào tập điều kiện hiệu chỉnh $do(T)$ tại thời điểm cấp học bổng nhập học**, ngăn chặn triệt để hiện tượng Conditioning on Post-Treatment Variables.

---

## 3. NOTEBOOK 2 & 3: KHÁM PHÁ CẤU TRÚC ĐỒ THỊ & PHÂN RÃ CƠ CHẾ TRUNG GIAN

### 3.1 Cơ chế Tác động Trực tiếp & Gián tiếp
*Đường dẫn tệp:* [fig4_mediation_path_decomposition.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig4_mediation_path_decomposition.png)

![Figure 4: Mediation Path Decomposition](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig4_mediation_path_decomposition.png)

#### Phân tích chi tiết:
Sơ đồ phân rã trung gian giải thích câu hỏi cốt lõi: *"Học bổng giúp sinh viên tốt nghiệp bằng con đường nào?"*
1. **Tổng hiệu ứng Can thiệp (Total Effect - TE):** $+21.61\%$ tăng tỷ lệ tốt nghiệp toàn diện.
2. **Hiệu ứng Trực tiếp Tự nhiên (Natural Direct Effect - NDE):**
   - Đạt **$+16.96\%$**, chiếm tới **$78.5\%$** tổng tác động của học bổng.
   - **Giải thích cơ chế kinh tế - xã hội:** Học bổng trực tiếp giải tỏa áp lực sinh kế, ngăn chặn nguy cơ phải nghỉ học để đi làm thêm kiếm sống hoặc bị đình chỉ học do nợ học phí. Đây là "chiếc phao cứu sinh" tài chính vận hành độc lập với năng lực học tập ban đầu.
3. **Hiệu ứng Gián tiếp Tự nhiên (Natural Indirect Effect - NIE):**
   - Đạt **$+4.65\%$**, đóng góp **$21.5\%$** tổng tác động.
   - **Giải thích cơ chế bồi dưỡng học thuật:**
     - Tác động $T \to M$: Nhận học bổng giúp sinh viên tích lũy thêm trung bình $\alpha = +0.86$ tín chỉ đỗ trong học kỳ 1 (do có thời gian chuyên tâm học tập và tinh thần an tâm).
     - Tác động $M \to Y$: Mỗi tín chỉ đỗ thêm ở học kỳ 1 chuyển hóa thành $\beta = +5.41\%$ tăng xác suất tốt nghiệp chung cuộc.
   - Kết luận: Học bổng vừa là công cụ an sinh tài chính trực tiếp ($78.5\%$), vừa là động lực tạo đà học thuật sớm ($21.5\%$).

---

## 4. NOTEBOOK 4: HIỆU ỨNG CAN THIỆP KHÔNG ĐỒNG NHẤT TRÊN QUẦN THỂ

### 4.1 Forest Plot Phân tích CATE theo Phân nhóm Sinh viên
*Đường dẫn tệp:* [fig5_heterogeneous_cate_forest.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig5_heterogeneous_cate_forest.png)

![Figure 5: Heterogeneous CATE Forest Plot](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig5_heterogeneous_cate_forest.png)

#### Phân tích chi tiết:
Biểu đồ Forest Plot trình bày ước lượng Hiệu ứng Can thiệp Cá thể hóa trung bình (Conditional Average Treatment Effect - CATE) kèm khoảng tin cậy $95\%$ qua từng phân tầng nhân khẩu học và kinh tế:
1. **Mức ATE Trung bình Quần thể (Baseline Benchmark):**
   - Hình thoi màu đỏ: $ATE = +22.3\% \ [20.3\%, 24.3\%]$.
2. **Bất đối xứng theo Nhóm tuổi (Age Disparity):**
   - **Sinh viên lớn tuổi ($>20$ tuổi):** $CATE = +28.0\% \ [24.5\%, 31.5\%]$.
   - **Sinh viên đúng tuổi ($\le 20$ tuổi):** $CATE = +16.1\% \ [13.2\%, 19.0\%]$.
   - Chênh lệch CATE giữa hai nhóm lên tới $+11.9\%$. Sinh viên lớn tuổi thường chịu gánh nặng tự trang trải sinh hoạt phí cao hơn nhiều, do đó sự hỗ trợ của học bổng mang lại giá trị giải phóng cận biên (marginal utility) lớn hơn gấp gần 1.74 lần.
3. **Điều kiện Nhà ở (Living Conditions):**
   - Sinh viên sống cùng gia đình (Local): $CATE = +25.9\% \ [22.1\%, 29.7\%]$.
   - Sinh viên thuê trọ xa nhà (Displaced): $CATE = +18.1\% \ [15.3\%, 20.9\%]$.
4. **Tình trạng Nợ học phí (Debtor Status):**
   - Sinh viên có nợ học phí (`Debtor = 1`): $CATE = +20.3\% \ [15.8\%, 24.7\%]$.
   - Sinh viên không nợ (`Debtor = 0`): $CATE = +21.3\% \ [19.1\%, 23.6\%]$.
   - Hiệu ứng tích cực hiện diện bền vững trên cả hai nhóm, chứng minh học bổng có tác dụng phổ quát không phụ thuộc vào tiền sử nợ nần.

---

## 5. NOTEBOOK 5: ĐÁNH GIÁ ĐỘ NHẠY THIÊN LỆCH BIẾN ẨN & TỐI ƯU HÓA CHÍNH SÁCH CAN THIỆP

### 5.1 Đồ thị Đường đồng mức Độ nhạy Cinelli-Hazlett
*Đường dẫn tệp:* [fig1_cinelli_hazlett_contour.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig1_cinelli_hazlett_contour.png)

![Figure 1: Cinelli-Hazlett Sensitivity Contour](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig1_cinelli_hazlett_contour.png)

#### Phân tích chi tiết:
Biểu đồ đánh giá mức độ vững chãi của ước lượng nhân quả trước nguy cơ vi phạm giả định "Không có biến ẩn gây nhiễu" (Unconfoundedness):
1. **Giá trị Hiệu ứng Cơ sở:** $\hat{\tau} = +0.2268$.
2. **Chỉ số Vững chãi (Robustness Value - RV):**
   - $RV_{q=1} = 19.1\%$ (tại điểm tròn đỏ có hộp ghi chú nổi bật).
   - **Ý nghĩa toán học:** Để kéo hiệu ứng học bổng từ $+0.2268$ về bằng $0$ (đường nét đứt màu đỏ $\hat{\tau}_{\text{adj}} = 0.00$), một biến ẩn chưa quan sát $U$ (ví dụ: động lực học tập nội tại, chỉ số IQ, hoặc hoàn cảnh khó khăn đột xuất) phải giải thích được **đồng thời ít nhất $19.1\%$** phương sai phần dư của xác suất nhận học bổng ($R^2_{T \sim U \mid Z}$) VÀ phương sai phần dư của xác suất tốt nghiệp ($R^2_{Y \sim U \mid Z}$).
3. **So chuẩn với các Biến quan sát thực tế (Observed Benchmark Confounders):**
   - Biến đóng học phí đúng hạn (`Tuition fees - Z5`): $R^2_{T \sim U} \approx 6.5\%$, $R^2_{Y \sim U} \approx 10.4\%$.
   - Biến độ tuổi nhập học (`Age - Z1`): $R^2_{T \sim U} \approx 3.0\%$, $R^2_{Y \sim U} \approx 4.2\%$.
   - Biến nợ học phí (`Debtor - Z4`), điểm đầu vào (`Z2`), trình độ mẹ (`Z3`): đều có $R^2 < 2.5\%$.
   - **Kết luận khoa học:** Toàn bộ các biến quan sát mạnh nhất trong thực tế đều nằm sâu bên dưới đường ranh giới triệt tiêu. Một biến ẩn chưa quan sát phải mạnh **gấp gần 2 lần biến đóng học phí ($Z_5$)** ở cả hai trục thì mới làm thay đổi kết luận. Do đó, kết luận học bổng tăng tỷ lệ tốt nghiệp là cực kỳ vững chãi.

---

### 5.2 Đường cong Học Chính sách Can thiệp dưới Ràng buộc Ngân sách
*Đường dẫn tệp:* [fig2_policy_learning_curve.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig2_policy_learning_curve.png)

![Figure 2: Optimal Causal Policy Learning Curve](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig2_policy_learning_curve.png)

#### Phân tích chi tiết:
Biểu đồ mô phỏng tỷ lệ tốt nghiệp toàn trường dự kiến khi triển khai các chính sách phân bổ học bổng khác nhau theo từng hạn mức ngân sách từ $0\%$ đến $100\%$:
1. **Đường cong Lồi tối ưu Nhân quả (CATE Optimal - Đường xanh lá đậm):**
   - Thể hiện tính quy luật năng suất biên giảm dần (diminishing marginal returns): thuật toán chính sách ưu tiên cấp học bổng cho các sinh viên có giá trị CATE cao nhất trước (nhóm lớn tuổi, hoàn cảnh khó khăn). Do đó, độ dốc ban đầu cực kỳ lớn.
2. **Điểm Can thiệp tại Ngân sách Thực tế ($26.7\%$ - Đường nét đứt màu đỏ dọc):**
   - Tương đương với quy mô $969$ suất học bổng hiện có của trường đại học.
   - **Chính sách Thực tế Lịch sử (Factual Allocation - Ngôi sao đỏ):** Tỷ lệ tốt nghiệp thực tế chỉ đạt **$60.85\%$**.
   - **Chính sách Tối ưu Nhân quả (CATE Optimal Policy - Vòng tròn xanh):** Đạt tỷ lệ tốt nghiệp dự kiến lên tới **$65.37\%$**.
   - **Thành quả tối ưu:** Tăng vọt thêm **$+4.52\%$ tỷ lệ tốt nghiệp**, tương đương với **$+164$ tân cử nhân tốt nghiệp mỗi khóa** mà **CHI PHÍ NGÂN SÁCH PHÁT SINH LÀ $0$ ĐỒNG** (chỉ cần tái cơ cấu thứ tự ưu tiên đối tượng nhận học bổng dựa trên mô hình nhân quả).
   - Vượt trội hoàn toàn so với chính sách Cấp theo nhu cầu nợ (`Need-based Debtor` - $60.2\%$) và Cấp ngẫu nhiên (`Random Allocation` - $55.6\%$).

---

## 6. NOTEBOOK SYNTHESIS: ĐÁNH GIÁ MA TRẬN ĐA NHÂN TỐ DỰ BÁO & NHÂN QUẢ

### 6.1 Đánh đổi Brier - AUROC vs Tính Nhất quán ATE
*Đường dẫn tệp:* [fig3_factorial_matrix_tradeoff.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig3_factorial_matrix_tradeoff.png)

![Figure 3: Factorial Tradeoff Matrix](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig3_factorial_matrix_tradeoff.png)

#### Phân tích chi tiết:
Đánh giá 4 cấu hình kiến trúc theo thiết kế thực nghiệm $2 \times 2$ Factorial Design:
- **B0:** Baseline Causal Bayesian Network (Tabular CPD + Domain DAG)
- **H1:** Hybrid Neural CPD (Neural MLP CPD + Domain DAG)
- **H2:** Hybrid Neural Discovery (Tabular CPD + NOTEARS Learned DAG)
- **H3:** Fully Neural Hybrid (Neural MLP CPD + NOTEARS Learned DAG)

1. **Đánh giá Năng lực Dự báo Ngoài mẫu (Panel A - Trái):**
   - Trục hoành là Brier Score (càng thấp càng tốt / độ hiệu chuẩn xác suất), trục tung là AUROC (càng cao càng tốt / năng lực phân loại).
   - Mô hình Baseline Tabular B0 và H2 có Brier score $\approx 0.1719$, AUROC $\approx 0.7955$.
   - Các cấu hình tích hợp mạng nơ-ron đa tầng MLP (H1, H3) dịch chuyển mạnh sang góc trên bên trái: Brier score giảm xuống $\approx 0.1667$ (cải thiện hiệu chuẩn xác suất) và AUROC tăng lên $\approx 0.8053$.
2. **Đánh giá Tính Nhất quán Ước lượng Can thiệp do(T) (Panel B - Phải):**
   - Biểu đồ thanh so sánh giá trị ATE kèm khoảng tin cậy $95\%$:
     - B0 (Baseline CBN): $+0.2370 \pm 0.012$
     - H1 (Neural CPD): $+0.2221 \pm 0.014$
     - H2 (Neural Discovery): $+0.2370 \pm 0.012$
     - H3 (Fully Neural): $+0.2228 \pm 0.014$
   - Cả 4 cấu hình đều hội tụ chặt chẽ quanh đường tham chiếu ATE $\sim +0.23$.
   - **Đóng góp khoa học:** Khẳng định nguyên lý bất biến nhân quả: Việc hiện đại hóa hàm phân phối điều kiện sang Neural MLP giúp tăng cường độ chính xác dự báo mà hoàn toàn không làm sai lệch hay suy thoái giá trị ước lượng can thiệp nhân quả $do(T)$.

---

## 7. NOTEBOOK 6: THỰC NGHIỆM QUY MÔ LỚN 76.518 DÒNG & TÍNH TỔNG QUÁT HÓA NGOẠI SUY

### 7.1 Kiểm định Mật độ Phân bố Dữ liệu Lớn
*Đường dẫn tệp:* [fig6_uci_vs_kaggle_distributions.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig6_uci_vs_kaggle_distributions.png)

![Figure 6: UCI vs Kaggle Distributions](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig6_uci_vs_kaggle_distributions.png)

#### Phân tích chi tiết:
Biểu đồ 4 góc kiểm định độ tin cậy của tập dữ liệu tổng hợp Kaggle S4E6 quy mô $76.518$ dòng so với tập gốc UCI:
1. **Panel A (Nhãn mục tiêu Target):**
   - Graduate: $49.9\%$ (UCI) vs $47.4\%$ (Kaggle).
   - Dropout: $32.1\%$ (UCI) vs $33.1\%$ (Kaggle).
   - Enrolled: $18.0\%$ (UCI) vs $19.5\%$ (Kaggle).
   - Sai số tỷ lệ tuyệt đối $< 2.5\%$.
2. **Panel B (Mật độ Tuổi nhập học):**
   - Cả hai phân phối trùng khít với đỉnh nhọn tại độ tuổi 18-20 và phần đuôi dài kéo sang phải tới 60 tuổi.
3. **Panel C (Mật độ Điểm tuyển sinh):**
   - Thuật toán CTGAN làm mịn phân bố điểm liên tục trong khoảng 100 đến 160 mà không làm méo mó các phân vị.
4. **Panel D (Tỷ lệ Can thiệp & Biến gây nhiễu chính):**
   - Học bổng ($T$): $26.7\%$ vs $27.5\%$ (lệch $0.8\%$).
   - Nợ học phí ($Z_4$): $11.5\%$ vs $8.0\%$ (lệch $3.5\%$).
   - Học phí đúng hạn ($Z_5$): $86.5\%$ vs $87.2\%$ (lệch $0.7\%$).
   - Mẹ học đại học ($Z_3$): $76.2\%$ vs $73.8\%$ (lệch $2.4\%$).
   - Kết luận: Môi trường dữ liệu 76k dòng giữ vững toàn bộ các mối tương quan điều kiện của bài toán nhân quả.

---

### 7.2 Bền vững Nhân quả qua Chuyển dịch Phân phối & Đánh giá OOD
*Đường dẫn tệp:* [fig7_cross_dataset_causal_generalization.png](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig7_cross_dataset_causal_generalization.png)

![Figure 7: Cross Dataset Causal Generalization](file:///e:/01_Research_Study_Master/04_Probabilistic_GrModel/artifacts/figures/fig7_cross_dataset_causal_generalization.png)

#### Phân tích chi tiết:
Đây là kiểm nghiệm mang tính đột phá của toàn bộ công trình: Đánh giá khả năng tổng quát hóa ngoài phân phối (Out-Of-Distribution - OOD Transfer) của Mạng Bayes Nhân quả.
1. **Tính Bền vững ATE qua Chuyển dịch Dữ liệu (Panel A - Trái):**
   - **UCI 697 (Baseline B0):** $+0.2370$.
   - **Kaggle S4E6 (B0):** $+0.3040 \pm 0.0022$ ($95\%$ Bootstrap CI: $[+0.3016, +0.3061]$).
   - **Kaggle S4E6 (H2 - Learned DAG):** $+0.3031$.
   - **Thực nghiệm Chuyển giao UCI $\to$ Kaggle Transfer:**
     - Mô hình CBN được huấn luyện trên **chỉ 3.630 sinh viên thật** của trường đại học Bồ Đào Nha, sau đó thực hiện suy luận can thiệp $do(T)$ trên toàn bộ **61.578 sinh viên** của tập dữ liệu Kaggle.
     - Kết quả ước lượng đạt **$ATE = +0.2143$**, hầu như trùng khớp với ước lượng nội bộ gốc của UCI ($+0.2153$).
2. **Khả năng Tổng quát hóa Ngoài Phân phối OOD (Panel B - Phải):**
   - Biểu đồ không gian Brier Score (trục hoành) vs AUROC (trục tung):
     - **Mô hình UCI Nội bộ:** AUROC $= 0.804$, Brier $= 0.176$.
     - **Mô hình Chuyển giao UCI $\to$ Kaggle (Chấm xanh lá):** AUROC đạt **$0.8537$**, Brier score đạt **$0.1539$**.
     - Điểm chấm xanh lá nằm sát cạnh mô hình huấn luyện nội bộ trực tiếp trên Kaggle (Chấm cam: AUROC $0.8550$, Brier $0.1472$).
   - **Ý nghĩa khoa học:** Trong khi các mô hình học máy tương quan thông thường thường bị suy giảm hiệu năng nghiêm trọng khi kiểm thử OOD (Distribution Shift / Out-of-Distribution Degradation), Mô hình Đồ thị Nhân quả CBN bảo toàn trọn vẹn sức mạnh phân loại và hiệu chuẩn xác suất. Điều này chứng minh Đồ thị Causal DAG đã học được cơ chế sinh dữ liệu bất biến (Invariant Causal Mechanism) chứ không chỉ nắm bắt các tương quan giả tạm thời.

---

## 8. TỔNG KẾT & KHUYẾN NGHỊ CHÍNH SÁCH

Qua việc đọc và phân tích toàn diện 16 tài sản hình ảnh từ 6 notebook thực nghiệm, công trình rút ra 4 kết luận khoa học và khuyến nghị thực tiễn:

1. **Hiệu quả Nhân quả Xác thực:** Học bổng làm tăng tỷ lệ tốt nghiệp từ $+21.4\%$ đến $+30.4\%$ với độ tin cậy thống kê tuyệt đối (Bootstrap $p < 0.0001$).
2. **Cơ chế Tác động Chủ đạo:** $78.5\%$ tác động đến từ việc trực tiếp giải tỏa áp lực sinh kế và nợ nần, trong khi $21.5\%$ tác động đến từ việc bồi dưỡng đà tích lũy tín chỉ học tập ở học kỳ 1.
3. **Giá trị Tối ưu hóa Ngân sách:** Thay đổi tiêu chí phân bổ học bổng dựa trên ước lượng CATE (ưu tiên sinh viên lớn tuổi và sinh viên có rủi ro tài chính cao) giúp nhà trường tăng thêm $+4.52\%$ tỷ lệ tốt nghiệp (+164 sinh viên tốt nghiệp/năm) với chi phí ngân sách bổ sung bằng $0$ đồng.
4. **Tính Bền vững & Chuyển giao Cao:** Cấu trúc mạng Bayes nhân quả bảo toàn hiệu năng khi chuyển giao giữa các tập dữ liệu có quy mô chênh lệch hơn 17 lần ($4.4k \to 76.5k$), chứng minh tính ưu việt của tiếp cận Mô hình Đồ thị có Xác suất kết hợp Nhân quả luận.
