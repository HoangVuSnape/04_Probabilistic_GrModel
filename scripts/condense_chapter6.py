from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\6_Conclusion")

def update_c6():
    # 6_1
    c6_1 = r"""\section{Synthesis of Findings and Hypothesis Verification}
\label{sec:concl_summary}

This research developed, implemented, and empirically audited a Hybrid Neural-Causal Probabilistic Graphical Modeling framework for educational policy intervention. Table \ref{tab:hypotheses_resolution} summarizes the formal resolution of our foundational research questions and hypotheses.

\begin{table}[htbp]
    \centering
    \small
    \caption{Systematic Verification of Research Questions and Scientific Hypotheses.}
    \label{tab:hypotheses_resolution}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lp{4.2cm}p{6.8cm}c}
        \toprule
        \textbf{Question} & \textbf{Formal Hypothesis} & \textbf{Empirical Finding} & \textbf{Resolution} \\
        \midrule
        \textbf{RQ1 / H1} & Continuous Neural CPDs resolve CPT cell explosion and improve calibration. & Neural CPDs ($H_1$) reduce Brier error from 0.1701 to 0.1648 and raise AUROC to 0.8082 without sparse-cell degradation. & \textbf{Supported} \\
        \midrule
        \textbf{RQ2 / H2} & Observational models are confounded; Pearl $do$-calculus isolates true ATE. & Raw observational gap (+33.9\%) overstates causal impact (+23.1\%) by 31.86\%. True ATE is +23.09\% across folds. & \textbf{Supported} \\
        \midrule
        \textbf{RQ3 / H3} & Mediation separates academic momentum from direct financial relief. & $NDE = +16.95\%$ (78.47\%) and $NIE = +4.65\%$ (21.53\% via Sem 1 credits). Robustness Value $RV_{q=1} = 23.40\%$. & \textbf{Supported} \\
        \midrule
        \textbf{RQ4 / H4} & CATE-guided policy yields major graduation welfare gains at fixed budget. & Reallocating 969 aid slots via CATE increases graduation from 60.85\% to 65.37\% (+164 graduates annually at \$0 added cost). & \textbf{Supported} \\
        \bottomrule
    \end{tabular}%
    }
\end{table}

\subsection{Theoretical Takeaways}
Our findings demonstrate that: (1) pairing an inspectable DAG with local neural probability surfaces eliminates the historical trade-off between expressive deep representation and transparent causal inference; (2) domain-enforced temporal forbidden masking is an absolute prerequisite to prevent continuous structure discovery (NOTEARS) from committing reverse causality; and (3) structural causal models provide inherent robustness against out-of-distribution covariate shifts ($|\Delta ATE| = 0.0010$ across 76,518 Kaggle instances).
"""
    with open(base_dir / "6_1_Summary_of_Findings.tex", "w", encoding="utf-8") as f:
        f.write(c6_1)

    # 6_2
    c6_2 = r"""\section{Institutional Policy Recommendations}
\label{sec:concl_recommendations}

The empirical findings translate into four actionable institutional directives for higher education administrators:

\begin{enumerate}
    \item \textbf{Transition from Correlational Scoring to Counterfactual Simulation:} Deprecate predictive "early warning" risk scores that only report dropout probability. Deploy causal graphical engines capable of evaluating counterfactual interventions ($P(Y \mid do(T=t), \mathbf{x})$) before public funds are committed.
    \item \textbf{Overhaul Merit-Only Allocation (Targeting Vulnerability):} Shift away from allocating aid solely on entry grades ($Z_2$), which subsidizes affluent students who would graduate regardless. Prioritize non-traditional older students ($>20$ years) and financially indebted students, who exhibit treatment multipliers of $1.72\times$ and $2.03\times$.
    \item \textbf{Couple Financial Aid with Early Academic Tutoring:} Because $21.53\%$ of the scholarship's causal effect is mediated through first-semester course completions ($T \to M_1 \to Y$), financial aid must be paired with proactive coaching during Semester 1 to protect academic momentum.
    \item \textbf{Strict Intervention Window (End of Semester 1):} By the end of Semester 1, course approval records provide definitive identification of students in the dropout spiral. Intervening after Semester 2 is largely ineffective because student departure has already crystallized.
\end{enumerate}
"""
    with open(base_dir / "6_2_Institutional_Policy_Recommendations.tex", "w", encoding="utf-8") as f:
        f.write(c6_2)

    # 6_3
    c6_3 = r"""\section{Limitations and Threats to Validity}
\label{sec:concl_limitations}

Four primary scientific limitations boundary the current findings:
\begin{enumerate}
    \item \textbf{Single-Institution Context:} The primary empirical cohort originates from the Polytechnic Institute of Portalegre. While OOD transfer across 76,518 Kaggle records demonstrated structural stability, deployment in other national jurisdictions requires local recalibration of financial thresholds.
    \item \textbf{Coarse Administrative Features:} Public registrar records bin complex socioeconomic realities into categorical proxies (e.g., parental education levels), omitting liquid savings or informal family remittances.
    \item \textbf{Unobserved Psychological Confounders:} Although the Cinelli-Hazlett audit proves an omitted confounder would need $RV = 23.40\%$ ($2.25\times$ first-semester grades) to nullify findings, unmeasured clinical depression or acute bereavement cannot be ruled out.
    \item \textbf{Discrete Administrative Epochs:} The current framework models discrete semesters ($\mathcal{E}_0 \dots \mathcal{E}_5$). Real campus disengagement is a continuous-time dynamic process marked by missed lectures and declining LMS engagement.
\end{enumerate}
"""
    with open(base_dir / "6_3_Limitations_and_Threats_to_Validity.tex", "w", encoding="utf-8") as f:
        f.write(c6_3)

    # 6_4
    c6_4 = r"""\section{Future Research Directions}
\label{sec:concl_future}

This monograph maps four technical frontiers for ongoing research:
\begin{enumerate}
    \item \textbf{Continuous-Time Dynamic SCMs:} Extending discrete epochs into continuous-time jump process models driven by LMS clickstreams to infer the optimal timestamp $t^*$ for automated advisor nudges.
    \item \textbf{Graph Neural Diffusion over Peer Networks:} Relaxing SUTVA to capture dormitory and study-group spillover effects using relational Graph Neural Networks.
    \item \textbf{Multi-Semester Contextual Bandits:} Formulating multi-stage scholarship allocation as a Causal Markov Decision Process to dynamically adjust aid across all eight semesters.
    \item \textbf{LLM-Assisted Causal Graph Discovery:} Employing reasoning Large Language Models to automatically extract candidate DAG constraints from university academic charters and faculty bylaws.
\end{enumerate}
"""
    with open(base_dir / "6_4_Future_Research_Directions.tex", "w", encoding="utf-8") as f:
        f.write(c6_4)

    print("Successfully condensed Chapter 6 files!")

if __name__ == '__main__':
    update_c6()
