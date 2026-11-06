import re
from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex")

def fix_2_2():
    p = base_dir / "2_chapters" / "2_Background" / "2_2_Structural_Causal_Models.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Table causal hierarchy
    text = re.sub(r'\\begin\{tabularx\}\{\\textwidth\}\{llp\{4\.5cm\}X\}',
                  r'\\small\n\\begin{tabularx}{\\textwidth}{p{2.2cm}p{3.2cm}p{4.2cm}X}', text)
    
    # Equation ATE
    ate_pat = re.compile(r'\\begin\{equation\}\s*ATE = \\mathbb\{E\}\[Y \\mid do\(T=1\)\] - \\mathbb\{E\}\[Y \\mid do\(T=0\)\] = .*?\\label\{eq:ate_definition\}\s*\\end\{equation\}', re.DOTALL)
    ate_repl = r'''\\begin{equation}
\\begin{aligned}
    ATE &= \\mathbb{E}[Y \\mid do(T=1)] - \\mathbb{E}[Y \\mid do(T=0)] \\\\
        &= \\sum_{\\mathbf{z}} \\left[ P(Y=1 \\mid T=1, \\mathbf{z}) - P(Y=1 \\mid T=0, \\mathbf{z}) \\right] P(\\mathbf{z}).
\\end{aligned}
\\label{eq:ate_definition}
\\end{equation}'''
    # We do a direct string replace if possible
    old_eq = r"""\begin{equation}
    ATE = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)] = \sum_{\mathbf{z}} \left[ P(Y=1 \mid T=1, \mathbf{z}) - P(Y=1 \mid T=0, \mathbf{z}) \right] P(\mathbf{z}).
    \label{eq:ate_definition}
\end{equation}"""
    new_eq = r"""\begin{equation}
\begin{aligned}
    ATE &= \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)] \\
        &= \sum_{\mathbf{z}} \left[ P(Y=1 \mid T=1, \mathbf{z}) - P(Y=1 \mid T=0, \mathbf{z}) \right] P(\mathbf{z}).
\end{aligned}
\label{eq:ate_definition}
\end{equation}"""
    if old_eq in text:
        text = text.replace(old_eq, new_eq)
    
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 2_2")

def fix_4_2():
    p = base_dir / "2_chapters" / "4_Experiments" / "4_2_HeldOut_Predictive_Performance.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    clean_table_a = r"""\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{5pt}
    \caption{Table A: Held-Out Predictive Performance Across 5-Fold Cross-Validation (Mean $\pm$ Std).}
    \label{tab:table_a_prediction}
    \begin{tabular}{llccccc}
        \toprule
        \textbf{Model} & \textbf{Graph Topology} & \textbf{AUROC} $\uparrow$ & \textbf{Macro F1} $\uparrow$ & \textbf{Brier Score} $\downarrow$ & \textbf{Precision} $\uparrow$ & \textbf{Recall} $\uparrow$ \\
        \midrule
        \textbf{$B_0$ (Baseline)} & Domain DAG & $0.8001 \pm 0.0135$ & $0.7308 \pm 0.0202$ & $0.1701 \pm 0.0067$ & $0.7902 \pm 0.0255$ & $0.7916 \pm 0.0145$ \\
        \textbf{$H_1$ (Neural CPD)} & Domain DAG & $\mathbf{0.8082 \pm 0.0159}$ & $\mathbf{0.7374 \pm 0.0165}$ & $\mathbf{0.1648 \pm 0.0071}$ & $0.7910 \pm 0.0222$ & $\mathbf{0.8072 \pm 0.0181}$ \\
        \textbf{$H_2$ (Learned DAG)} & Discovered DAG & $0.6428 \pm 0.0000$ & $0.5790 \pm 0.0000$ & $0.2145 \pm 0.0000$ & $\mathbf{0.8630 \pm 0.0000}$ & $0.3795 \pm 0.0000$ \\
        \textbf{$H_3$ (Fully Neural)} & Discovered DAG & $0.8048 \pm 0.0122$ & $0.7147 \pm 0.0225$ & $0.1667 \pm 0.0070$ & $0.7885 \pm 0.0194$ & $0.7960 \pm 0.0210$ \\
        \bottomrule
    \end{tabular}
\end{table}"""
    text = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_table_a, text, flags=re.DOTALL)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 4_2")

def fix_4_3():
    p = base_dir / "2_chapters" / "4_Experiments" / "4_3_Causal_Interventions_and_ATE.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    clean_table_b = r"""\begin{table}[htbp]
    \centering
    \small
    \setlength{\tabcolsep}{6pt}
    \caption{Table B: Standardized Causal Interventions and Average Treatment Effects ($ATE$).}
    \label{tab:table_b_intervention}
    \begin{tabular}{llcccc}
        \toprule
        \textbf{Model} & \textbf{Graph Topology} & \textbf{$\mathbb{E}[Y_0]$} & \textbf{$\mathbb{E}[Y_1]$} & \textbf{Causal ATE} & \textbf{95\% CI} \\
        \midrule
        \textbf{$B_0$ (Baseline)} & Domain DAG & $0.5499 \pm 0.0114$ & $0.7863 \pm 0.0087$ & $\mathbf{+0.2363 \pm 0.0061}$ & $[+0.2240, +0.2487]$ \\
        \textbf{$H_1$ (Neural CPD)} & Domain DAG & $0.5535 \pm 0.0121$ & $0.7844 \pm 0.0087$ & $\mathbf{+0.2309 \pm 0.0097}$ & $[+0.2223, +0.2397]$ \\
        \textbf{$H_2$ (Learned DAG)} & Discovered DAG & $0.5163 \pm 0.0000$ & $0.8600 \pm 0.0000$ & $+0.3437 \pm 0.0000$ & $[+0.3437, +0.3437]$ \\
        \textbf{$H_3$ (Fully Neural)} & Discovered DAG & $0.5518 \pm 0.0105$ & $0.7746 \pm 0.0091$ & $\mathbf{+0.2228 \pm 0.0101}$ & $[+0.2150, +0.2310]$ \\
        \bottomrule
    \end{tabular}
\end{table}"""
    text = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_table_b, text, flags=re.DOTALL)

    eq_clean = r"""\begin{align}
    \tau_{\text{obs}} &= P(Y=1 \mid T=1) - P(Y=1 \mid T=0) = 81.3\% - 47.4\% = \mathbf{+33.9\%}, \label{eq:naive_gap} \\
    \tau_{\text{causal}} &= \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)] = 78.4\% - 55.4\% = \mathbf{+23.1\%}. \label{eq:causal_ate}
\end{align}"""
    text = re.sub(r'\\begin\{align\}.*?\\end\{align\}', lambda m: eq_clean, text, flags=re.DOTALL)

    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 4_3")

def fix_4_4():
    p = base_dir / "2_chapters" / "4_Experiments" / "4_4_DoWhy_Refutations_and_Computational_Efficiency.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    clean_table_c = r"""\begin{table}[htbp]
    \centering
    \small
    \caption{Table C: DoWhy Refutation Audit, Empirical Stability Spread, and Hardware Runtimes.}
    \label{tab:table_c_robustness}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Model} & \textbf{Placebo Effect} & \textbf{Subset Effect} & \textbf{Paired Spread} & \textbf{$\Delta\text{ATE}$ vs $B_0$} & \textbf{Runtime} & \textbf{Audit Status} \\
        \midrule
        \textbf{$B_0$} & $+0.0003$ & $+0.2341$ & $0.0137$ & --- & 0.27s & Passed (Robust) \\
        \textbf{$H_1$} & $+0.0080$ & $+0.2347$ & $0.0231$ & $-0.0054$ & 27.57s & Passed (Robust) \\
        \textbf{$H_2$} & $-0.0044$ & $+0.2328$ & $\mathbf{0.0000}$ & $+0.1074$ & 2.20s & Passed (Robust) \\
        \textbf{$H_3$} & $+0.0062$ & $+0.2285$ & $0.0184$ & $-0.0135$ & 31.40s & Passed (Robust) \\
        \bottomrule
    \end{tabular}
\end{table}"""
    text = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_table_c, text, flags=re.DOTALL)
    text = re.sub(r'\\includegraphics\[width=[0-9.]+\\textwidth\]\{figs/fig3_factorial_matrix_tradeoff\.png\}',
                  r'\\includegraphics[width=0.60\\textwidth]{figs/fig3_factorial_matrix_tradeoff.png}', text)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 4_4")

def fix_5_1():
    p = base_dir / "2_chapters" / "5_Advanced_Analysis" / "5_1_Alternative_Academic_Outcomes.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    clean_tab_alt = r"""\begin{table}[htbp]
    \centering
    \small
    \caption{Causal Treatment Effects of Scholarship Grants Across Multi-Dimensional Academic Outcomes.}
    \label{tab:alternative_outcomes}
    \begin{tabularx}{\textwidth}{L c c c c c}
        \toprule
        \textbf{Outcome Dimension} & \textbf{Cohort} & \textbf{Causal Estimand} & \textbf{Estimated ATE} & \textbf{95\% Conf. Interval} & \textbf{$p$-value} \\
        \midrule
        \textbf{Continuous Sem 2 GPA} & $N = 3,630$ & $\mathbb{E}[Y_1] - \mathbb{E}[Y_0]$ & \textbf{+1.478 pts} & $[+1.093, +1.863]$ & $6.69 \times 10^{-14}$ \\
        \textbf{Credit Completion Ratio} & $N = 3,630$ & $\mathbb{E}[Y_1] - \mathbb{E}[Y_0]$ & \textbf{+14.50\%} & $[+11.80\%, +17.20\%]$ & $1.58 \times 10^{-25}$ \\
        \textbf{Full Retention (AIPW)} & $N = 4,424$ & Doubly Robust AIPW & \textbf{+16.87\%} & $[+13.79\%, +19.73\%]$ & $< 1.0 \times 10^{-10}$ \\
        \bottomrule
    \end{tabularx}
\end{table}"""
    text = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_tab_alt, text, flags=re.DOTALL)

    new_eq = r"""\begin{equation}
\begin{aligned}
    \widehat{ATE}_{\text{GPA}} &= \mathbb{E}[Y_{\text{GPA}} \mid do(T=1)] - \mathbb{E}[Y_{\text{GPA}} \mid do(T=0)] \\
    &= \mathbf{+1.478 \text{ pts}} \quad (95\% \text{ CI}: [1.093, 1.863], \, p = 6.69 \times 10^{-14}).
\end{aligned}
\label{eq:ate_gpa}
\end{equation}"""
    text = re.sub(r'\\begin\{equation\}\s*\\widehat\{ATE\}_\{.*?\\label\{eq:ate_gpa\}\s*\\end\{equation\}',
                  lambda m: new_eq, text, flags=re.DOTALL)

    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 5_1")

def fix_5_3():
    p = base_dir / "2_chapters" / "5_Advanced_Analysis" / "5_3_Heterogeneous_Treatment_Effects_CATE.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    clean_tab_cate = r"""\begin{table}[htbp]
    \centering
    \small
    \caption{Heterogeneous Treatment Effects (CATE) Across Demographic and Socioeconomic Subgroups.}
    \label{tab:heterogeneous_cate}
    \begin{tabularx}{\textwidth}{Lccccc}
        \toprule
        \textbf{Demographic Subgroup} & \textbf{$N$} & \textbf{$\mathbb{E}[Y_0]$} & \textbf{$\mathbb{E}[Y_1]$} & \textbf{CATE} & \textbf{Risk Ratio} \\
        \midrule
        \textbf{Age: Non-Traditional ($> 20$ yrs)} & 1,550 & 38.72\% & 66.74\% & \textbf{+28.02\%} & \textbf{1.72$\times$} \\
        \textbf{Age: Traditional ($\le 20$ yrs)} & 2,080 & 68.55\% & 84.63\% & +16.08\% & 1.23$\times$ \\
        \midrule
        \textbf{Finance: Debtor (Active Arrears)} & 413 & 19.74\% & 40.01\% & \textbf{+20.27\%} & \textbf{2.03$\times$} \\
        \textbf{Finance: Solvent (No Arrears)} & 3,217 & 60.02\% & 81.37\% & +21.35\% & 1.36$\times$ \\
        \midrule
        \textbf{Housing: Living Locally} & 1,637 & 48.33\% & 74.22\% & \textbf{+25.88\%} & 1.54$\times$ \\
        \textbf{Housing: Displaced / Renting} & 1,993 & 61.41\% & 79.01\% & +17.60\% & 1.29$\times$ \\
        \bottomrule
    \end{tabularx}
\end{table}"""
    text = re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}', lambda m: clean_tab_cate, text, flags=re.DOTALL)
    text = re.sub(r'\\includegraphics\[width=[0-9.]+\\textwidth\]\{figs/fig5_heterogeneous_cate_forest\.png\}',
                  r'\\includegraphics[width=0.60\\textwidth]{figs/fig5_heterogeneous_cate_forest.png}', text)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 5_3")

def fix_5_6():
    p = base_dir / "2_chapters" / "5_Advanced_Analysis" / "5_6_Large_Scale_Kaggle_Transferability_OOD.tex"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove separate figures
    text = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:uci_kaggle_dist\}\s*\\end\{figure\}', '', text, flags=re.DOTALL)
    combined_fig = r"""\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig6_uci_vs_kaggle_distributions.png}
        \caption{UCI vs. Kaggle Covariate Densities}
        \label{fig:uci_kaggle_dist}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig7_cross_dataset_causal_generalization.png}
        \caption{Zero-Shot Transfer Invariance}
        \label{fig:cross_generalization}
    \end{subfigure}
    \caption{Cross-Dataset Empirical Validation on Kaggle S4E6 ($N=76,518$). (a) Marginal covariate distribution shifts; (b) Empirical proof of modular causal invariance ($|\Delta\text{ATE}| = 0.0010$).}
    \label{fig:cross_dataset_combined}
\end{figure}"""
    text = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:cross_generalization\}\s*\\end\{figure\}',
                  lambda m: combined_fig, text, flags=re.DOTALL)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("Fixed 5_6")

if __name__ == '__main__':
    fix_2_2()
    fix_4_2()
    fix_4_3()
    fix_4_4()
    fix_5_1()
    fix_5_3()
    fix_5_6()
