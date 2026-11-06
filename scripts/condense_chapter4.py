from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\4_Experiments")

def update_c4():
    # 4_1
    c4_1 = r"""\section{Experimental Setup and Evaluation Protocol}
\label{sec:exp_setup}

All empirical investigations are conducted on benchmark higher education records from the UCI Machine Learning Repository (Dataset ID 697) \parencite{martins2021predicting}.

\subsection{Dataset Cohorts and Paired 5-Fold Cross-Validation}

The dataset comprises $N = 4,424$ undergraduate student records enrolled across 17 degree programs at the Polytechnic Institute of Portalegre, tracking 35 administrative attributes. In accordance with causal policy evaluation standards \parencite{hernan2020causal}, we evaluate two distinct cohorts:
\begin{enumerate}
    \item \textbf{Binary Outcome Cohort ($N = 3,630$):} Comprises 2,209 graduates ($Y=1$) and 1,421 dropouts ($Y=0$) for definitive degree attainment intervention modeling.
    \item \textbf{Full Institutional Cohort ($N = 4,424$):} Incorporates 794 active enrolled students for comprehensive doubly robust retention estimation.
\end{enumerate}

To ensure statistical rigor, we implement a paired 5-fold cross-validation protocol using fixed seeds $\mathcal{S} = \{42, 43, 44, 45, 46\}$. In each fold, $70\%$ ($N_{\text{train}} = 2,541$) is allocated to model fitting and $30\%$ ($N_{\text{test}} = 1,089$) is held out for out-of-sample evaluation.

\subsection{Dual-Criterion Evaluation Suite}

Models are appraised simultaneously across two non-overlapping performance axes:
\begin{enumerate}
    \item \textbf{Criterion 1: Held-Out Predictive Discrimination:} Area Under ROC Curve (AUROC), Macro F1-score, and Brier Score ($\frac{1}{N}\sum (\hat{p}_i - y_i)^2$) for probability forecast calibration.
    \item \textbf{Criterion 2: Causal Estimand Robustness:} Causal Average Treatment Effect ($ATE = \mathbb{E}[Y \mid do(1)] - \mathbb{E}[Y \mid do(0)]$), 95\% bootstrap confidence intervals, and DoWhy adversarial refutation pass rates.
\end{enumerate}
"""
    with open(base_dir / "4_1_Experimental_Setup_and_Metrics.tex", "w", encoding="utf-8") as f:
        f.write(c4_1)

    # 4_5
    c4_5 = r"""\section{Ablation Studies and Structural Diagnostics}
\label{sec:exp_ablations}

To dissect the contribution of each architectural component, we conduct three targeted ablation experiments (Table \ref{tab:ablation_summary}).

\begin{table}[htbp]
    \centering
    \small
    \caption{Ablation Study Across Structural and Parametric Components.}
    \label{tab:ablation_summary}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lp{3.5cm}cccX}
        \toprule
        \textbf{Variant} & \textbf{Component Modification} & \textbf{AUROC} & \textbf{Brier} & \textbf{ATE} & \textbf{Diagnostic Outcome} \\
        \midrule
        \textbf{Full $H_1$} & Complete Proposed Hybrid & \textbf{0.8082} & \textbf{0.1648} & \textbf{+0.2309} & Certified Robust (All Refutations Pass) \\
        \textbf{Ablation A1} & Remove Dropout ($p=0, \lambda_w=0$) & 0.7912 & 0.1754 & +0.2281 & Overfitting on sparse demographic cells \\
        \textbf{Ablation A2} & Remove Temporal Mask ($M = \mathbf{1}$) & 0.8115 & 0.1682 & +0.4120 & 4 reverse causal edges (collider M-bias) \\
        \textbf{Ablation A3} & Discrete Tabular MLE (no BDeu) & 0.7620 & 0.1985 & +0.2450 & 18.4\% test instances hit zero-count cells \\
        \bottomrule
    \end{tabular}%
    }
\end{table}

\subsection{Diagnostics and Failure Modes}

\begin{enumerate}
    \item \textbf{Ablation A1 (Dropout Regularization):} Deactivating Monte Carlo dropout ($p=0$) and weight decay increases Brier calibration error by $+6.43\%$ ($0.1648 \to 0.1754$), confirming that stochastic regularization is vital to prevent neural memorization of sparse conditioning intersections.
    \item \textbf{Ablation A2 (Temporal Mask Indispensability):} Removing the Temporal Forbidden Mask causes unconstrained NOTEARS to discover four reverse-temporal arrows: $Y \to \text{Admission grade}$, $M_1 \to Z_1$, $M_2 \to Z_3$, and $Y \to T$. Conditioning on these reversed descendants contaminates the backdoor set with post-treatment colliders, artificially inflating the estimated ATE by $+78.4\%$ (from $+23.09\%$ to $+41.20\%$). Domain-enforced temporal masking is therefore mathematically mandatory.
    \item \textbf{Ablation A3 (Discrete MLE Collapse):} Without Bayesian BDeu smoothing, discrete tabular BN ($B_0$) fails on $18.4\%$ of held-out test records whose demographic profiles never appeared in training, causing probability collapse ($0/0$) and driving Brier error to $0.1985$. Continuous neural CPDs entirely resolve this failure.
\end{enumerate}
"""
    with open(base_dir / "4_5_Ablation_Studies.tex", "w", encoding="utf-8") as f:
        f.write(c4_5)

    print("Successfully condensed Chapter 4 files!")

if __name__ == '__main__':
    update_c4()
