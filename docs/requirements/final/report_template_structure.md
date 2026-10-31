# KHUNG CẤU TRÚC BÁO CÁO KHOA HỌC CUỐI KỲ (LATEX TEMPLATE OUTLINE)
> Dành cho việc soạn thảo bài báo Cuối kỳ môn **Mô hình đồ thị có xác suất (IT701100)**  
> **Yêu cầu kỹ thuật:** Biên dịch từ mã nguồn LaTeX sang file PDF hoàn chỉnh (theo Slide 49, chấp nhận file PDF biên dịch từ LaTeX hoặc file `.docx` học thuật chuẩn mực).  
> **Lưu ý Turnitin:** Bài viết sẽ được kiểm duyệt qua **Turnitin Check AI & Plagiarism**. Văn phong cần tự viết học thuật, mạch lạc, có trích dẫn chuẩn mực và đính kèm bản tuyên bố sử dụng AI minh bạch.  
> **Tham chiếu Rubric 10 điểm:** Slide 45–50 & 78 (`HK1_2026_IT701100_Mô hình đồ thị có xác suất_N01.pdf` - 80 slides).

---

```latex
\documentclass[10pt,journal,compsoc]{IEEEtran}
% Hoặc sử dụng: \documentclass[runningheads]{llncs} hoặc \documentclass{article}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{url}
\usepackage{hyperref}

\begin{document}

\title{Evaluating Neural Enhancements in Causal Bayesian Networks: An Empirical Study on Tabular Educational Data}

\author{Nguyen Van A (First Author), Tran Van B (Co-Author) \\
Department of Information Technology, Master of Science Program \\
Instructor: Nguyen Chi Thien, Ph.D.}

\maketitle

% =========================================================================
% 2.8. ABSTRACT (1.0 ĐIỂM) [CLO4, CLO5] - SLIDE 49, 78
% =========================================================================
\begin{abstract}
Estimating interventional causal effects from observational tabular data remains a cornerstone of informed policy making. While classical Causal Bayesian Networks (CBNs) provide rigorous graphical identification and transparent uncertainty estimation via Pearl's do-calculus, they rely on discrete parameterizations that struggle with high-dimensional interactions. In this work, we present a controlled empirical benchmark comparing a classical discrete CBN baseline (B0) against two principled deep learning enhancements: neural conditional probability tables (H1) and gradient-based neural structure discovery with domain temporal masks (H2). We evaluate all models across five paired random seeds on the UCI Student Performance dataset under a strictly shared train/validation/test protocol. Predictive capability is evaluated against held-out test labels, whereas causal intervention risks and Average Treatment Effects (ATE) are standardized across a common test baseline reference cohort ($C_{\text{test}}$). All models are subjected to formal backdoor identification and DoWhy refutation checks (placebo treatment and data subset refuters). Our findings demonstrate that while neural CPDs enhance predictive likelihood and discriminatory power, causal effect stability critically depends on valid domain temporal constraints rather than unconstrained neural optimization. We discuss the empirical implications of positivity violations, structural unidentifiability, and computational tradeoffs.
\end{abstract}

\begin{IEEEkeywords}
Causal Bayesian Networks, Deep Learning, do-calculus, Average Treatment Effect, Neural Causal Discovery, DAG-GNN, Refutation, Reproducibility.
\end{IEEEkeywords}

% =========================================================================
% 2.1. INTRODUCTION (1.0 ĐIỂM) [CLO1, CLO2] - SLIDE 45, 78
% =========================================================================
\section{Introduction}
\label{sec:intro}
% Gợi ý nội dung theo Slide 45, 78:
% 1. Đặt vấn đề thực tiễn: Nhu cầu ra quyết định dựa trên can thiệp chính sách (intervention) thay vì tương quan quan sát (correlation).
% 2. Nguyên lý khoa học: Khuyến nghị chính sách bắt buộc phải dựa trên toán tử can thiệp P(Y | do(T=t)); việc suy diễn can thiệp từ xác suất quan sát P(Y | T=t) là sai lầm nghiêm trọng.
% 3. Tại sao CBN là mốc tham chiếu nền tảng: Cung cấp ngôn ngữ đồ thị chuẩn tắc, phân rã điều kiện trong suốt và định danh hình thức (formal identification).
% 4. Động lực thử nghiệm các thành phần Deep Learning (H1, H2): Liệu mạng nơ-ron có cải thiện được độ khớp dữ liệu và cấu trúc đồ thị, hay gây ra quá khớp (overfitting) và thiếu ổn định nhân quả?
% 5. Tóm tắt các đóng góp chính và lộ trình bài báo.

% =========================================================================
% 2.2. RELATED WORK (1.0 ĐIỂM) [CLO1, CLO2] - SLIDE 45, 78
% =========================================================================
\section{Related Work}
\label{sec:related_work}
% Gợi ý nội dung theo Slide 45, 78:
% 1. Causal Bayesian Networks & Pearl's Causal Hierarchy: Khác biệt giữa Quan sát (L1), Can thiệp (L2), Phản thực tế (L3).
% 2. Deep Learning for Causal Mechanisms (H1): Sử dụng mạng MLP/nơ-ron ước lượng P(Y | Parents(Y)) với hàm mất mát Binary Cross-Entropy.
% 3. Gradient-based Continuous Causal Discovery (H2):
%    - Chuyển bài toán tổ hợp rời rạc sang tối ưu gradient liên tục (NOTEARS - Zheng et al., DAG-GNN - Yu et al.).
%    - Tính vi phân của điều kiện phi chu trình: h(W) = Tr(e^{W \circ W}) - d = 0.
%    - Vai trò sống còn của ma trận mặt nạ ràng buộc miền (Domain Mask M): W_eff = M \odot W để triệt tiêu cạnh ngược thời gian.
% 4. Thư viện chuẩn mực và môi trường thực nghiệm: pgmpy, DoWhy, PyTorch.

% =========================================================================
% 2.3. QUANTITATIVE PROBLEM STATEMENT (1.0 ĐIỂM) [CLO1, CLO2, CLO4] - SLIDE 45, 54, 78
% =========================================================================
\section{Quantitative Problem Statement}
\label{sec:problem_statement}
% Gợi ý nội dung theo Slide 45, 54, 78:
% 1. Không gian bài toán định lượng: Nêu rõ 1 câu định danh quần thể mục tiêu, biến can thiệp T \in {0, 1}, biến kết quả Y \in {0, 1}, thời điểm can thiệp.
% 2. Bảng từ điển biến số (Variable Dictionary Table): Tên biến, kiểu mã hóa, thời điểm đo lường, vai trò nhân quả (Treatment / Outcome / Baseline Confounder), tỷ lệ missing.
% 3. Mục tiêu can thiệp định lượng (Causal Estimand):
%    \tau = \mathbb{E}[Y \mid \text{do}(T=1)] - \mathbb{E}[Y \mid \text{do}(T=0)]
% 4. Ba giai đoạn nhân quả rạch ròi:
%    - Causal Discovery: Học đồ thị DAG G = (V, E).
%    - Formal Identification: Áp dụng tiêu chuẩn Backdoor để biến đổi đại lượng can thiệp thành biểu thức quan sát:
%      P(Y=1 \mid \text{do}(T=t)) = \sum_{c} P(Y=1 \mid T=t, C=c) P(C=c)
%    - Statistical Estimation: Ước lượng xác suất từ dữ liệu mẫu.
% 5. Tuyên bố các giả định nhận diện: No unmeasured confounding, Consistency, Positivity/Overlap (0 < P(T=1 | C=c) < 1).

% =========================================================================
% 2.4. PROPOSED METHODS AND MODEL DEFINITIONS (2.0 ĐIỂM) [CLO3, CLO4] - SLIDE 46, 57-63, 78
% =========================================================================
\section{Methods and Model Definitions}
\label{sec:methods}
% Gợi ý nội dung theo Slide 46, 57-63, 78:
% 1. B0 (Baseline CBN):
%    - Thuật toán học cấu trúc truyền thống (pgmpy HillClimbSearch / PC / GES) trên tập Train với ràng buộc thời gian.
%    - Khớp tham số CPD bằng Bayesian smoothing với prior BDeu (equivalent sample size = 5).
% 2. H1 (Neural Mechanism on Fixed DAG):
%    - Cố định DAG B0 cho từng run, thay thế CPD của Y bằng mạng nơ-ron nhận đầu vào DUY NHẤT là Parents(Y).
%    - Kiến trúc: MLP, hàm kích hoạt ReLU, Dropout, BCE loss, Validation early stopping.
%    - Rời rạc hóa tổ hợp nút cha để tạo bảng CPD chuẩn hóa lắp lại vào mạng Bayes.
% 3. H2 (Neural Causal Discovery with Domain Mask):
%    - Thuật toán tối ưu gradient liên tục (DAG-GNN / NOTEARS).
%    - Ma trận kề W, ma trận mặt nạ miền M: W_eff = M \odot W, đặt M_{ij} = 0 nếu cạnh vi phạm thời gian.
%    - Hàm phạt chu trình h(W_eff) = 0, ngưỡng lọc cạnh (thresholding), và quy tắc sửa chu trình (cycle repair).
%    - Khớp CPD non-neural bằng cùng phương pháp Bayesian smoothing như B0.
% 4. Công thức tính can thiệp chuẩn hóa (Slide 61):
%    - Với mỗi cá nhân i trên tập Test, tính q_m(t, C_i) = P_m(Y=1 | T=t, C_i).
%    - Chuẩn hóa trên tập tham chiếu C_test: risk_m(t) = (1/n_test) \sum_{i \in TEST} q_m(t, C_i).
%    - ATE_m = risk_m(1) - risk_m(0). (Tuyệt đối không dùng nhãn Y_test để tính hiệu ứng).
% 5. Phương pháp ước lượng độ bất định: 95% Confidence Interval (Bootstrap / Normal / Delta method).

% =========================================================================
% 2.5. FAIR COMPARISON AND EXPERIMENTAL DESIGN (1.0 ĐIỂM) [CLO3, CLO4] - SLIDE 47, 55-56, 68-71, 78
% =========================================================================
\section{Fair Comparison and Experimental Design}
\label{sec:exp_design}
% Gợi ý nội dung theo Slide 47, 55-56, 68-71, 78:
% 1. Kiểm toán dữ liệu UCI (Data Audit): Nguồn gốc, URL, checksum, lưu đồ lọc dữ liệu (eligibility flow), kiểm toán thời điểm đo lường và kiểm tra tính bao phủ (treatment overlap).
% 2. Giao thức chia tập dữ liệu:
%    - Phân chia Train/Validation/Test (vd: 70%/15%/15%).
%    - Tiền xử lý (imputation, scaling, discretization) CHỈ fit trên tập Train.
%    - Tập Validation dùng để chọn ngưỡng đồ thị, dừng sớm; tập Test ĐÓNG BĂNG cho đánh giá cuối.
% 3. Giao thức chạy lặp ghép cặp (Paired Runs Protocol):
%    - Chạy trên ÍT NHẤT 5 LẦN CHẠY GHÉP CẶP (paired runs >= 5) dùng chung chỉ số dữ liệu.
%    - Báo cáo giá trị trung bình \pm độ lệch chuẩn và chênh lệch ghép cặp có dấu (H1 - B0, H2 - B0).
% 4. Cấu hình kiểm định bác bỏ DoWhy (DoWhy Refutation Protocol):
%    - Placebo treatment refuter: Hoán vị ngẫu nhiên nhãn can thiệp T.
%    - Data subset refuter: Lấy mẫu lại 80% tập dữ liệu để đánh giá độ nhạy mẫu.

% =========================================================================
% 2.6. EXPERIMENTAL RESULTS (1.0 ĐIỂM) [CLO3, CLO4] - SLIDE 48, 74-75, 78
% =========================================================================
\section{Experimental Results}
\label{sec:results}
% Bắt buộc trình bày đủ 3 bảng kết quả chuẩn theo Slide 74-75:

% -------------------------------------------------------------------------
% BẢNG A: SO SÁNH NĂNG LỰC DỰ BÁO NGOÀI MẪU (SLIDE 74)
% -------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{Table A: Standard Prediction Comparison on Held-Out Test Data (Mean $\pm$ Std across 5 paired runs)}
\label{tab:table_a}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{Graph Specification} & \textbf{AUROC} & \textbf{Macro-F1} & \textbf{Brier Score} & $\mathbf{\Delta_{\text{Model}-B_0}\text{ (AUROC)}}$ & \textbf{Fit Time (s)} & \textbf{Query Time (s)} & \textbf{Hardware} \\
\midrule
\textbf{B0} & Domain DAG & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & --- & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ & CPU \\
\textbf{H1} & Fixed B0 DAG (Neural CPD) & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $+0.000$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ & GPU \\
\textbf{H2} & Learned DAG (Neural Discovery) & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $+0.000$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ & GPU \\
\bottomrule
\end{tabular}
\end{table*}

% -------------------------------------------------------------------------
% BẢNG B: ĐÁNH GIÁ HIỆU ỨNG CAN THIỆP CHUẨN HÓA TRÊN TẬP TEST (SLIDE 75)
% -------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{Table B: Standardized Intervention Estimates on Common Test Reference Cohort ($C_{\text{test}}$)}
\label{tab:table_b}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Reference Cohort ($n_{\text{test}}$)} & \textbf{Backdoor Adjustment Set} & $\mathbf{P(Y=1 \mid \text{do}(T=0))}$ & $\mathbf{P(Y=1 \mid \text{do}(T=1))}$ & $\mathbf{\widehat{\text{ATE}}}$ \textbf{(95\% CI)} & \textbf{CI Method / Status} \\
\midrule
\textbf{B0} & Test Set ($N=\dots$) & $\{Z_1, Z_2, \dots, Z_k\}$ & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $+0.000 \; [0.000, 0.000]$ & Bootstrap / Valid DAG \\
\textbf{H1} & Test Set ($N=\dots$) & $\{Z_1, Z_2, \dots, Z_k\}$ & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $+0.000 \; [0.000, 0.000]$ & Bootstrap / Valid DAG \\
\textbf{H2} & Test Set ($N=\dots$) & Identified Formula & $0.000 \pm 0.000$ & $0.000 \pm 0.000$ & $+0.000 \; [0.000, 0.000]$ & Bootstrap / Learned DAG \\
\bottomrule
\end{tabular}
\end{table*}

% -------------------------------------------------------------------------
% BẢNG C: TÍNH VỮNG BÁC BỎ VÀ CHI PHÍ TÍNH TOÁN (SLIDE 75)
% -------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{Table C: Robustness Refutations, Run Stability, and Computational Cost}
\label{tab:table_c}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Placebo Refuter Effect} & \textbf{Data Subset Refuter Effect} & \textbf{Paired Effect Spread} & $\mathbf{\Delta_{\text{ATE}}(\text{Model}-B_0)}$ & \textbf{Total Time (s)} & \textbf{Hardware Specification} \\
\midrule
\textbf{B0} & $0.000 \; (p > 0.05)$ & Stable ($\Delta < 5\%$) & $0.000$ & --- & $0.00$ s & CPU (Intel Core i7 / Colab) \\
\textbf{H1} & $0.000 \; (p > 0.05)$ & Stable ($\Delta < 5\%$) & $0.000$ & $+0.000$ & $0.00$ s & GPU (NVIDIA T4) \\
\textbf{H2} & $0.000 \; (p > 0.05)$ & Stable ($\Delta < 5\%$) & $0.000$ & $+0.000$ & $0.00$ s & GPU (NVIDIA T4) \\
\bottomrule
\end{tabular}
\end{table*}

% Thảo luận chi tiết về kết quả:
% - So sánh AUROC/F1 giữa B0, H1, H2.
% - So sánh giá trị ATE chuẩn hóa và khoảng tin cậy.
% - Đánh giá tính đạt của 2 bài kiểm thử bác bỏ DoWhy.

% =========================================================================
% 2.7. CONCLUSIONS AND LIMITATIONS (1.0 ĐIỂM) [CLO4, CLO5] - SLIDE 48, 78, 79
% =========================================================================
\section{Conclusions and Limitations}
\label{sec:conclusion}
% Gợi ý nội dung theo Slide 48, 78, 79:
% 1. Đúc kết khoa học: Phân tích thành phần nào thực sự mang lại hiệu quả, khi nào B0 đã đủ tốt.
% 2. Báo cáo trung thực các phát hiện rỗng (null findings) và sự cố thực nghiệm theo Slide 79:
%    - Xử lý ô giá trị thưa / vi phạm tính bao phủ (positivity).
%    - Đồ thị H2 sinh chu trình hoặc vi phạm trật tự thời gian.
%    - Lỗi ô xác suất bằng 0 khi dùng Maximum Likelihood.
%    - Hiện tượng quá khớp (overfitting) của mạng nơ-ron H1.
% 3. Thảo luận các giới hạn nhân quả cốt lõi: Nguy cơ biến nhiễu ẩn (unmeasured confounding) và tính khái quát hóa sang các trường học khác.

% =========================================================================
% 2.9. QUESTIONS AND ANSWERS (DEFENSE PREPARATION - 1.0 ĐIỂM) [CLO1-CLO5] - SLIDE 49, 78
% =========================================================================
\section{Questions and Answers (Defense Preparation)}
\label{sec:qa}
% Soạn thảo sẵn câu trả lời biện giải khoa học cho các câu hỏi trọng yếu:
% - Q1: Tại sao lại áp đặt cấu trúc thời gian và ma trận mặt nạ cạnh cấm M trong H2?
% - Q2: Tại sao kết quả dự báo tốt (AUROC cao) không chứng minh được độ chính xác của hiệu ứng nhân quả?
% - Q3: Tại sao vượt qua hai bài kiểm tra bác bỏ DoWhy vẫn chưa thể khẳng định 100% không có biến nhiễu ẩn?
% - Q4: Tại sao hiệu ứng can thiệp phải được chuẩn hóa trên cùng tập tham chiếu C_test thay vì tính trung bình riêng rẽ?

% =========================================================================
% AI-USE DISCLOSURE (BẮT BUỘC THEO QUY ĐỊNH HỌC THUẬT - SLIDE 49, 78)
% =========================================================================
\section*{AI-Use Disclosure}
We transparently disclose the use of AI tools during the execution of this research. Generative AI tools (e.g., ChatGPT, Gemini) were utilized strictly for code syntax suggestions, language polishing, and exploring domain constraint candidates. All domain constraints, DAG directions, mathematical formulations, and experimental conclusions were independently validated and verified by the authors. No text was generated unreviewed, and all empirical results were computed directly from executable Python pipelines.

% =========================================================================
% TÀI LIỆU THAM KHẢO CHUẨN MỰC (SLIDE 52, 80)
% =========================================================================
\begin{thebibliography}{00}
\bibitem{pearl2009} J. Pearl, \emph{Causality: Models, Reasoning, and Inference}, 2nd ed. Cambridge University Press, 2009.
\bibitem{dowhy} A. Sharma and E. Kiciman, ``DoWhy: An End-to-End Library for Causal Inference,'' \emph{arXiv:2011.04216}, 2020.
\bibitem{pgmpy} A. Ankan and A. Panda, ``pgmpy: Probabilistic Graphical Models using Python,'' in \emph{Proc. of the 14th Python in Science Conf.}, 2015.
\bibitem{notears} X. Zheng, B. Aragam, P. Ravikumar, and E. P. Xing, ``DAGs with NO TEARS: Continuous Optimization for Structure Learning,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2018.
\bibitem{daggnn} Y. Yu, J. Chen, T. Gao, and M. Yu, ``DAG-GNN: DAG Structure Learning with Graph Neural Networks,'' in \emph{Proc. of the 36th Int. Conf. on Machine Learning (ICML)}, 2019.
\bibitem{uci} UCI Machine Learning Repository, ``Student Performance Data Set,'' 2014. [Online]. Available: \url{https://archive.ics.uci.edu/dataset/320/student+performance}.
\end{thebibliography}

% =========================================================================
% PHỤ LỤC: MỞ RỘNG TÙY CHỌN (OPTIONAL EXTENSIONS - SLIDE 64-67, 80)
% =========================================================================
\appendices
\section{Optional Extensions: Ablation Studies and Synthetic Verification}
\label{sec:appendix_ablations}
% Trình bày các kết quả mở rộng tùy chọn (nếu có thực hiện):
% 1. Mô hình kết hợp H1+H2 (Neural CPD trên Learned DAG).
% 2. Nghiên cứu bóc tách có kiểm soát A1-A3:
%    - A1 (từ H1): Bỏ Dropout (rate = 0).
%    - A2 (từ H2): Gỡ bỏ 1 ràng buộc cấm cạnh thời gian trong M.
%    - A3 (từ B0): Thay Bayesian smoothing bằng Maximum Likelihood thuần túy.
% 3. Thí nghiệm Synthetic Tutoring Benchmark (chân lý ATE = 0.300) và mô phỏng DoWhy GCM.

\end{document}
```
