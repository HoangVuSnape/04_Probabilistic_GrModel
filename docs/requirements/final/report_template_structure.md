# KHUNG CẤU TRÚC BÁO CÁO KHOA HỌC CUỐI KỲ (LATEX TEMPLATE OUTLINE)
> Dành cho việc soạn thảo bài báo Cuối kỳ môn **Mô hình đồ thị có xác suất (IT701100)**  
> **Yêu cầu kỹ thuật:** Biên dịch từ mã nguồn LaTeX sang file PDF hoàn chỉnh.  
> **Lưu ý Turnitin:** Bài viết sẽ được kiểm duyệt qua **Turnitin Check AI & Plagiarism**. Văn phong cần tự viết học thuật, mạch lạc, có trích dẫn chuẩn mực và đính kèm bản tuyên bố sử dụng AI minh bạch.

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

\title{Causal Effect Estimation via Deep Neural Enhancements in Causal Bayesian Networks: An Empirical Study on Tabular Data}

\author{Nguyen Van A (First Author), Tran Van B (Co-Author) \\
Department of Information Technology, Master of Science Program \\
Instructor: Nguyen Chi Thien, Ph.D.}

\maketitle

% =========================================================================
% 2.8. ABSTRACT (1.0 ĐIỂM)
% =========================================================================
\begin{abstract}
Estimating interventional causal effects from observational tabular data remains a central challenge in data-driven decision making. While classical Causal Bayesian Networks (CBNs) provide rigorous, interpretable graph-based identification via Pearl's do-calculus, they often struggle with non-linear functional mechanisms and combinatorial structure learning. In this work, we investigate a systematic causal benchmarking framework comparing a baseline CBN (B0) against two deep learning (DL) hybrid extensions: neural parameter estimation (H1) and gradient-based neural structure discovery with domain constraints (H2), as well as a fully hybrid variant (H1+H2). We evaluate all models across five paired random seeds on both a synthetic benchmark with known population ground truth ($\text{ATE} = 0.30$) and the UCI Student Performance dataset. Furthermore, three strictly controlled ablations (A1–A3) are performed to isolate the isolated contributions of neural regularization, domain acyclicity masking, and probability smoothing. Our findings reveal that while neural CPDs enhance predictive fit, faithful causal estimation critically depends on domain temporal constraints and formal backdoor identification rather than unconstrained neural discovery. We validate all estimates against two DoWhy refutation checks (placebo treatment and data subset refuters) and discuss the fundamental limits imposed by unmeasured confounding.
\end{abstract}

\begin{IEEEkeywords}
Causal Bayesian Networks, Deep Learning, do-calculus, Average Treatment Effect, Neural Causal Discovery, DAG-GNN, Refutation.
\end{IEEEkeywords}

% =========================================================================
% 2.1. INTRODUCTION (1.0 ĐIỂM)
% =========================================================================
\section{Introduction}
\label{sec:intro}
% Gợi ý nội dung:
% 1. Đặt vấn đề thực tiễn: Nhu cầu ra quyết định dựa trên can thiệp (intervention) thay vì tương quan quan sát (correlation).
% 2. Nhấn mạnh: Đưa ra khuyến nghị chính sách phải dựa trên đo lường can thiệp P(Y | do(T=t)), việc suy luận can thiệp từ quan sát thuần túy là sai lầm khoa học nghiêm trọng.
% 3. Tại sao CBN là mốc tham chiếu nền tảng: Cung cấp ngôn ngữ đồ thị chuẩn tắc, định danh hình thức (identification).
% 4. Động lực thử nghiệm các thành phần Deep Learning (H1, H2): Liệu mạng nơ-ron có cải thiện được việc ước lượng tham số và học cấu trúc hay gây ra quá khớp (overfitting)?
% 5. Tóm tắt các đóng góp chính của bài báo.

% =========================================================================
% 2.2. RELATED WORK (1.0 ĐIỂM)
% =========================================================================
\section{Related Work}
\label{sec:related_work}
% Gợi ý nội dung:
% 1. Causal Bayesian Networks & Pearl's Causal Hierarchy: Sự khác biệt giữa Association (L1), Intervention (L2), Counterfactual (L3).
% 2. Deep Learning for Causal Mechanisms (H1): Sử dụng mạng MLP/nơ-ron ước lượng P(Y | Parents(Y)).
% 3. Gradient-based Continuous Causal Discovery (H2):
%    - Chuyển bài toán tổ hợp rời rạc sang tối ưu gradient liên tục (NOTEARS - Zheng et al., DAG-GNN - Yu et al.).
%    - Tính vi phân của điều kiện phi chu trình: h(W) = Tr(e^{W \circ W}) - d = 0.
%    - Tầm quan trọng của ma trận mặt nạ ràng buộc miền (Domain Constraints Mask) để giải quyết tính tương đương Markov.
% 4. Thư viện chuẩn mực: pgmpy và DoWhy.

% =========================================================================
% 2.3. QUANTITATIVE PROBLEM STATEMENT (1.0 ĐIỂM)
% =========================================================================
\section{Quantitative Problem Statement}
\label{sec:problem_statement}
% Gợi ý nội dung:
% 1. Không gian bài toán: Quần thể mục tiêu, biến can thiệp T \in {0, 1}, biến kết quả Y \in {0, 1}, biến gây nhiễu Z.
% 2. Bảng từ điển biến số (Variable Dictionary Table): Tên biến, kiểu, thời điểm đo lường, vai trò nhân quả.
% 3. Mục tiêu can thiệp định lượng (Causal Estimand):
%    \tau = \mathbb{E}[Y \mid \text{do}(T=1)] - \mathbb{E}[Y \mid \text{do}(T=0)]
% 4. Ba giai đoạn nhân quả rạch ròi:
%    - Causal Discovery: Học đồ thị DAG G = (V, E).
%    - Formal Identification: Áp dụng tiêu chuẩn Backdoor để biến đổi đại lượng can thiệp thành xác suất quan sát:
%      P(Y=1 \mid \text{do}(T=t)) = \sum_{z} P(Y=1 \mid T=t, Z=z) P(Z=z)
%    - Statistical Estimation: Ước lượng xác suất từ dữ liệu mẫu.
% 5. Tuyên bố các giả định nhận diện: No unmeasured confounding, Consistency, Positivity (0 < P(T=t | Z=z) < 1).

% =========================================================================
% 2.4. PROPOSED METHODS AND ABLATION DEFINITIONS (2.0 ĐIỂM)
% =========================================================================
\section{Proposed Methods and Ablations}
\label{sec:methods}
% Gợi ý nội dung:
% 1. B0 (Baseline CBN):
%    - Domain DAG thiết kế từ tri thức chuyên gia.
%    - Ước lượng CPD bằng Bayesian smoothing với BDeu prior (equivalent sample size = 5).
%    - Thuật toán suy diễn Variable Elimination.
% 2. H1 (Neural Mechanism):
%    - Cố định DAG B0, thay CPD của Y bằng mạng nơ-ron nhận đầu vào duy nhất là Parents(Y).
%    - Chi tiết kiến trúc: Dense layers, hàm kích hoạt ReLU, Binary Cross-Entropy loss, Dropout(0.2).
% 3. H2 (Gradient-based Neural Structure Learning):
%    - Tối ưu hóa gradient ma trận kề W.
%    - Ma trận mặt nạ ràng buộc miền M: W_eff = M \odot W, đặt M_{ij} = 0 nếu cạnh vi phạm thời gian.
%    - Tái định danh công thức can thiệp bằng DoWhy và khớp CPD non-neural.
% 4. H1+H2 (Hybrid Extension - Mở rộng kết hợp):
%    - Sử dụng đồ thị đã học từ H2, tích hợp thêm cơ chế nơ-ron của H1.
% 5. Ba biến thể bóc tách A1–A3 (Nguyên tắc One change only):
%    - A1: H1 bỏ Dropout (dropout rate = 0).
%    - A2: H2 gỡ bỏ 1 ràng buộc cấm cạnh thời gian trong M.
%    - A3: B0 thay Bayesian smoothing bằng Maximum Likelihood thuần túy.

% =========================================================================
% 2.5. FAIR COMPARISON AND EXPERIMENTAL DESIGN (1.0 ĐIỂM)
% =========================================================================
\section{Experimental Design}
\label{sec:exp_design}
% Gợi ý nội dung:
% 1. Kiểm toán dữ liệu UCI (ID: 320 - Student Performance) hoặc dữ liệu thu thập thêm:
%    - Xử lý loại bỏ rò rỉ nhãn (bỏ điểm G3).
%    - Kiểm tra độ bao phủ (treatment overlap) giữa T và Z.
% 2. Giao thức chia tập và hạt giống:
%    - Chia Train/Val/Test (70%/15%/15%).
%    - Thực hiện trên ÍT NHẤT 5 HẠT GIỐNG NGẪU NHIÊN (seeds >= 5).
% 3. Bài toán mô phỏng chân lý (Synthetic Tutoring Benchmark):
%    - N = 20,000 mẫu, chân lý ATE = 0.30, P(Y=1|do(T=0))=0.30, P(Y=1|do(T=1))=0.60.
% 4. Cấu hình kiểm định bác bỏ DoWhy:
%    - Placebo treatment refuter (100 lần mô phỏng hoán vị nhãn T).
%    - Data subset refuter (100 lần mô phỏng lấy 80% mẫu con).
%    - Mô phỏng can thiệp GCM.

% =========================================================================
% 2.6. EXPERIMENTAL RESULTS (1.0 ĐIỂM)
% =========================================================================
\section{Experimental Results}
\label{sec:results}
% Bắt buộc trình bày 3 bảng kết quả chuẩn:

% BẢNG A: SO SÁNH B0, H1, H2, H1+H2 TRÊN DỮ LIỆU UCI
% \begin{table*}[t]
% \caption{Table A: Baseline and Hybrid Comparisons on UCI Dataset (Mean $\pm$ Std over 5 seeds)}
% ...
% \end{table*}

% BẢNG B: KẾT QUẢ ĐỐI SOÁT VỚI CHÂN LÝ MÔ PHỎNG (SYNTHETIC BENCHMARK)
% \begin{table}[t]
% \caption{Table B: Known-Truth Synthetic Benchmark Results (True ATE = 0.300)}
% ...
% \end{table}

% BẢNG C: KẾT QUẢ NGHIÊN CỨU BÓC TÁCH (ABLATION ANALYSIS)
% \begin{table}[t]
% \caption{Table C: Controlled Ablation Analysis ($\Delta = \text{Variant} - \text{Parent}$)}
% ...
% \end{table}

% Báo cáo kết quả kiểm thử bác bỏ DoWhy và thảo luận độ biến thiên.

% =========================================================================
% 2.7. CONCLUSIONS AND LIMITATIONS (1.0 ĐIỂM)
% =========================================================================
\section{Conclusions and Limitations}
\label{sec:conclusion}
% Gợi ý nội dung:
% 1. Kết luận khoa học: Đánh giá thành phần nơ-ron nào thực sự đóng góp giá trị, khi nào B0 đã đủ tốt.
% 2. Báo cáo trung thực các kết quả rỗng (null findings) hoặc trường hợp nơ-ron kém hơn baseline.
% 3. Thảo luận các giới hạn:
%    - Nguy cơ yếu tố gây nhiễu chưa quan sát được (unmeasured confounding).
%    - Độ không chắc chắn của đồ thị học được từ dữ liệu quan sát.
%    - Vấn đề thiếu dữ liệu / ô xác suất thưa.

% =========================================================================
% 2.9. QUESTIONS AND ANSWERS (DEFENSE PREPARATION - 1.0 ĐIỂM)
% =========================================================================
\section{Questions and Answers (Defense)}
\label{sec:qa}
% Soạn thảo sẵn câu trả lời biện giải cho các câu hỏi chất vấn:
% - Q1: Tại sao lại áp đặt chiều mũi tên trong B0 DAG?
% - Q2: Tại sao vượt qua bài kiểm tra bác bỏ DoWhy vẫn chưa thể khẳng định 100% quan hệ nhân quả?
% - Q3: Tại sao trong H2 bắt buộc phải áp dụng ma trận mặt nạ ràng buộc miền?

% =========================================================================
% AI-USE DISCLOSURE (BẮT BUỘC THEO QUY ĐỊNH HỌC THUẬT)
% =========================================================================
\section*{AI-Use Disclosure}
We transparently disclose the use of AI tools during this project...

% =========================================================================
% TÀI LIỆU THAM KHẢO
% =========================================================================
\begin{thebibliography}{00}
\bibitem{pearl2009} J. Pearl, \emph{Causality: Models, Reasoning, and Inference}, 2nd ed. Cambridge University Press, 2009.
\bibitem{dowhy} A. Sharma and E. Kiciman, ``DoWhy: An End-to-End Library for Causal Inference,'' \emph{arXiv:2011.04216}, 2020.
\bibitem{pgmpy} A. Ankan and A. Panda, ``pgmpy: Probabilistic Graphical Models using Python,'' in \emph{Proc. of the 14th Python in Science Conf.}, 2015.
\bibitem{notears} X. Zheng, et al., ``DAGs with NO TEARS: Continuous Optimization for Structure Learning,'' \emph{NeurIPS}, 2018.
\bibitem{daggnn} Y. Yu, et al., ``DAG-GNN: DAG Structure Learning with Graph Neural Networks,'' \emph{ICML}, 2019.
\end{thebibliography}

\end{document}
```
