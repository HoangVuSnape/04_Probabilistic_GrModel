from pathlib import Path

p_5_6 = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\5_Advanced_Analysis\5_6_Large_Scale_Kaggle_Transferability_OOD.tex")
with open(p_5_6, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(r"\caption{Cross-Dataset Empirical Validation on Kaggle S4E6 (=76,518$). (a) Marginal covariate distribution shifts; (b) Empirical proof of modular causal invariance (invariance gap $\le 0.0010$).}",
                    r"\caption{Cross-Dataset Empirical Validation on Kaggle S4E6 ($N=76,518$). (a) Marginal covariate distribution shifts; (b) Empirical proof of modular causal invariance (invariance gap $\le 0.0010$).}")

with open(p_5_6, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated 5_6 caption successfully")

p_5_1 = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\5_Advanced_Analysis\5_1_Alternative_Academic_Outcomes.tex")
with open(p_5_1, "r", encoding="utf-8") as f:
    text_51 = f.read()

clean_tab_alt = r"""\begin{table}[htbp]
    \centering
    \small
    \caption{Causal Treatment Effects of Scholarship Grants Across Multi-Dimensional Academic Outcomes.}
    \label{tab:alternative_outcomes}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Outcome Dimension} & \textbf{Cohort} & \textbf{Causal Estimand} & \textbf{Estimated ATE} & \textbf{95\% Conf. Interval} & \textbf{$p$-value} \\
        \midrule
        Continuous Sem 2 GPA & $N = 3,630$ & $\mathbb{E}[Y_1] - \mathbb{E}[Y_0]$ & \textbf{+1.478 pts} & $[+1.093, +1.863]$ & $6.69 \times 10^{-14}$ \\
        Credit Completion Ratio & $N = 3,630$ & $\mathbb{E}[Y_1] - \mathbb{E}[Y_0]$ & \textbf{+14.50\%} & $[+11.80\%, +17.20\%]$ & $1.58 \times 10^{-25}$ \\
        Full Retention (AIPW) & $N = 4,424$ & Doubly Robust AIPW & \textbf{+16.87\%} & $[+13.79\%, +19.73\%]$ & $< 1.0 \times 10^{-10}$ \\
        \bottomrule
    \end{tabular}
\end{table}"""

import re
text_51 = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_tab_alt, text_51, flags=re.DOTALL)
with open(p_5_1, "w", encoding="utf-8") as f:
    f.write(text_51)
print("Updated 5_1 table successfully")
