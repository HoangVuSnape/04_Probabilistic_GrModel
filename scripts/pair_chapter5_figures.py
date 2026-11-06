import re
from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\5_Advanced_Analysis")

def pair_figures():
    # 1. In 5_2: Remove standalone figure 4
    p_5_2 = base_dir / "5_2_Mediation_Analysis_Path_Decomposition.tex"
    with open(p_5_2, "r", encoding="utf-8") as f:
        t_5_2 = f.read()
    # Remove figure
    t_5_2 = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:mediation_decomp\}\s*\\end\{figure\}', '', t_5_2, flags=re.DOTALL)
    with open(p_5_2, "w", encoding="utf-8") as f:
        f.write(t_5_2)
    print("Cleaned 5_2")

    # 2. In 5_3: Replace standalone figure 5 with combined (fig4 + fig5)
    p_5_3 = base_dir / "5_3_Heterogeneous_Treatment_Effects_CATE.tex"
    with open(p_5_3, "r", encoding="utf-8") as f:
        t_5_3 = f.read()
    
    combined_med_cate = r"""\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig4_mediation_path_decomposition.png}
        \caption{Mediation Path Decomposition}
        \label{fig:mediation_decomp}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig5_heterogeneous_cate_forest.png}
        \caption{Subgroup CATE Estimates}
        \label{fig:cate_forest}
    \end{subfigure}
    \caption{Structural Causal Mechanisms and Subgroup Vulnerability. (a) Partition of Total Effect into Direct Financial Relief ($NDE = +16.95\%$) and Academic Momentum ($NIE = +4.65\%$); (b) Heterogeneous Treatment Effects (CATE) estimated via Causal Forest.}
    \label{fig:mechanisms_and_heterogeneity}
\end{figure}"""

    t_5_3 = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:cate_forest\}\s*\\end\{figure\}',
                   lambda m: combined_med_cate, t_5_3, flags=re.DOTALL)
    with open(p_5_3, "w", encoding="utf-8") as f:
        f.write(t_5_3)
    print("Updated 5_3 with combined mediation+cate figure")

    # 3. In 5_4: Remove standalone figure 1
    p_5_4 = base_dir / "5_4_Cinelli_Hazlett_Sensitivity_Contour.tex"
    with open(p_5_4, "r", encoding="utf-8") as f:
        t_5_4 = f.read()
    t_5_4 = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:cinelli_contour\}\s*\\end\{figure\}', '', t_5_4, flags=re.DOTALL)
    with open(p_5_4, "w", encoding="utf-8") as f:
        f.write(t_5_4)
    print("Cleaned 5_4")

    # 4. In 5_5: Replace standalone figure 2 with combined (fig1 + fig2)
    p_5_5 = base_dir / "5_5_Budget_Constrained_Policy_Optimization.tex"
    with open(p_5_5, "r", encoding="utf-8") as f:
        t_5_5 = f.read()

    combined_sens_policy = r"""\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig1_cinelli_hazlett_contour.png}
        \caption{Cinelli-Hazlett Sensitivity Contours}
        \label{fig:cinelli_contour}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figs/fig2_policy_learning_curve.png}
        \caption{Budget Policy Learning Curves}
        \label{fig:policy_curve}
    \end{subfigure}
    \caption{Sensitivity Auditing and Budget-Constrained Policy Optimization. (a) Critical contour for unobserved confounding ($RV_{q=1} = 23.40\%$); (b) Cohort graduation rate vs. award quota (+164 graduates under $\pi^*$).}
    \label{fig:sensitivity_and_policy}
\end{figure}"""

    t_5_5 = re.sub(r'\\begin\{figure\}\[htbp\].*?\\label\{fig:policy_curve\}\s*\\end\{figure\}',
                   lambda m: combined_sens_policy, t_5_5, flags=re.DOTALL)
    with open(p_5_5, "w", encoding="utf-8") as f:
        f.write(t_5_5)
    print("Updated 5_5 with combined sensitivity+policy figure")

    # 5. In main.tex: Replace \cleardoublepage with \clearpage
    main_p = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\main.tex")
    with open(main_p, "r", encoding="utf-8") as f:
        main_t = f.read()
    main_t = main_t.replace(r"\cleardoublepage", r"\clearpage")
    with open(main_p, "w", encoding="utf-8") as f:
        f.write(main_t)
    print("Updated main.tex")

if __name__ == '__main__':
    pair_figures()
