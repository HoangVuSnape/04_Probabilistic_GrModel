from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\1_Introduction")

def update_c1():
    # 1_1
    c1_1 = r"""\section{Reason for Choosing the Topic}

Student attrition in higher education represents a major socioeconomic challenge that impairs institutional sustainability, strains public educational budgets, and disrupts individual socioeconomic mobility \parencite{martins2021predicting}. Consequently, universities and government ministries allocate substantial fiscal capital toward student financial aid and scholarship programs. However, a foundational policy dilemma persists: \emph{Does awarding an operational scholarship ($T=1$) causally increase a student's degree completion probability ($Y=1$), or does the observed empirical correlation merely reflect pre-existing socioeconomic self-selection?}

Standard machine learning models (e.g., Gradient Boosted Trees, Deep Neural Classifiers) estimate observational conditional probabilities $P(Y=1 \mid T=t, \mathbf{X}=\mathbf{x})$. As proven by \textcite{pearl2009causality} and \textcite{hernan2020causal}, association does not imply causation:
\begin{equation}
    P(Y=1 \mid T=t) \neq P(Y=1 \mid do(T=t)).
    \label{eq:pearl_inequality}
\end{equation}
In university administrative registries, scholarship recipients systematically differ from non-recipients: they typically exhibit higher secondary school entrance grades ($Z_2$), greater family educational capital ($Z_1$), and better initial solvency ($Z_3$). Relying on associational correlations to guide policy risks severe misallocation of scarce public capital to students who would have graduated regardless of institutional aid.

Probabilistic Graphical Models (PGMs)---specifically Causal Bayesian Networks (CBNs) and Structural Causal Models (SCMs)---provide a formal graph-theoretic framework for causal identification \parencite{pearl2009causality, ankan2015pgmpy}. Through Pearl's $do$-calculus and graph surgery, incoming backdoor confounding arrows into $T$ are severed, identifying the interventional distribution:
\begin{equation}
    P(Y=1 \mid do(T=t)) = \sum_{\mathbf{z}} P(Y=1 \mid T=t, \mathbf{Z}=\mathbf{z}) P(\mathbf{Z}=\mathbf{z}).
    \label{eq:backdoor_formula_intro}
\end{equation}
However, canonical discrete Bayesian networks rely on tabular Conditional Probability Tables (CPTs), which suffer from combinatorial parameter explosion and sparse-cell failures when conditioning on multi-dimensional continuous covariates \parencite{koller2009pgm}. Conversely, deep neural networks provide flexible function approximation but lack causal semantics, frequently exploiting spurious correlations and collider biases \parencite{shi2019dragonnet, sharma2020dowhy}.

Developing a \textbf{Hybrid Neural-Causal Graphical Architecture} that integrates the structural identification of graphical models with the continuous density estimation of deep learning addresses this critical scientific challenge. This research establishes an end-to-end framework to isolate true causal effects, execute adversarial refutations, uncover mediation pathways, and optimize policy allocations under real-world fiscal constraints.
"""
    with open(base_dir / "1_1_Reason_for_choosing_the_topic.tex", "w", encoding="utf-8") as f:
        f.write(c1_1)

    # 1_2
    c1_2 = r"""\section{Research Objectives}

\subsection{General Objective}
The primary objective of this research is to develop, evaluate, and benchmark a principled \textbf{Hybrid Neural-Causal Probabilistic Graphical Modeling Framework} for educational policy intervention. By unifying Bayesian network structural identification with continuous neural CPD parameterization, this study estimates the true causal treatment effect of scholarship allocation on degree completion and derives optimal, budget-constrained institutional aid policies.

\subsection{Specific Objectives}
To achieve the general objective, this research undertakes four specific objectives:
\begin{enumerate}
    \item \textbf{SO1 (Structural Formulation and Identification):} Formalize a 7-node domain-grounded Causal Directed Acyclic Graph ($\mathcal{G}_{\text{domain}}$) and continuous structure learning pipeline (NOTEARS with Temporal Forbidden Masking) that provably satisfies backdoor admissibility for educational intervention $T$.
    \item \textbf{SO2 (Factorial Architecture Benchmarking):} Implement a controlled $2 \times 2$ factorial model matrix ($B_0$: Tabular+Domain, $H_1$: Neural+Domain, $H_2$: Tabular+Learned, $H_3$: Neural+Learned) to empirically isolate the distinct contributions of graph topology versus parameterization across 5-fold cross-validation.
    \item \textbf{SO3 (Adversarial Causal Robustness and Mediation):} Audit causal effect stability against placebo treatment permutations, random subset refutations, and Cinelli-Hazlett unobserved confounding bounds, while decomposing total effects into direct financial relief ($NDE$) and academic momentum ($NIE$).
    \item \textbf{SO4 (Policy Optimization and Out-of-Distribution Transfer):} Solve the budget-constrained policy allocation problem to maximize cohort degree completions at fixed institutional expenditure, and evaluate out-of-distribution causal invariance across 76,518 instances from the Kaggle S4E6 benchmark.
\end{enumerate}
"""
    with open(base_dir / "1_2_Research_objectives.tex", "w", encoding="utf-8") as f:
        f.write(c1_2)

    # 1_3
    c1_3 = r"""\section{Research Questions and Hypotheses}

This monograph addresses one overarching question and four specific research questions and paired hypotheses:

\paragraph{Overarching Question (RQ0):} Does financial scholarship provision ($T$) exert a genuine causal effect on university degree completion ($Y$) after adjusting for pre-enrollment confounders, through what structural mechanisms is this effect transmitted, and how can aid policies be optimized under institutional budget constraints?

\begin{enumerate}[leftmargin=*]
    \item \textbf{RQ1 (Predictive Calibration):} Does continuous neural CPD parameterization improve held-out predictive accuracy (AUROC, Macro F1) and probability calibration (Brier score) relative to discrete tabular CPTs?
    \begin{itemize}
        \item \textbf{Hypothesis H1:} Deep neural parameterization ($H_1, H_3$) will outperform discrete tabular baselines ($B_0, H_2$) in AUROC ($\Delta \ge +0.008$) and Brier calibration ($\Delta \le -0.005$) by interpolating across sparse multi-dimensional covariate cells.
    \end{itemize}

    \item \textbf{RQ2 (Confounding Distortion):} What proportion of the raw observational graduation gap between scholarship recipients and non-recipients is driven by spurious confounding?
    \begin{itemize}
        \item \textbf{Hypothesis H2:} The true causal Average Treatment Effect ($ATE$) will be substantially lower than the naive observational association gap, revealing an empirical confounding overestimation exceeding $25\%$.
    \end{itemize}

    \item \textbf{RQ3 (Mediation Transmission):} What fraction of the scholarship's causal effect is mediated through early academic momentum ($M_1$, first-semester approved credits) versus direct economic relief?
    \begin{itemize}
        \item \textbf{Hypothesis H3:} While direct economic relief ($NDE$) accounts for the majority of policy impact ($\ge 70\%$), early academic credit completion ($NIE$) mediates a statistically significant channel ($\ge 15\%$).
    \end{itemize}

    \item \textbf{RQ4 (Policy Welfare Gains):} Can individual treatment effect estimation (CATE) under finite institutional budgets enhance cohort degree attainment relative to historical merit-only allocation?
    \begin{itemize}
        \item \textbf{Hypothesis H4:} Allocating scholarships according to estimated CATE scores will yield a statistically significant graduation surge ($+100$ or more net additional graduates) at zero added fiscal cost.
    \end{itemize}
\end{enumerate}
"""
    with open(base_dir / "1_3_Research_questions_and_hypotheses.tex", "w", encoding="utf-8") as f:
        f.write(c1_3)

    # 1_4
    c1_4 = r"""\section{Research Subjects and Scope}

\subsection{Research Subjects}
The subjects of this research encompass two complementary domains:
\begin{enumerate}
    \item \textbf{Methodological Subject:} Formulation, implementation, and empirical auditing of Probabilistic Graphical Models for causal inference. This includes Causal Bayesian Networks, continuous structure discovery with temporal constraints (NOTEARS), deep neural parameterization (PyTorch Neural CPDs), semi-parametric doubly robust estimation (AIPW), and Cinelli-Hazlett sensitivity algebra.
    \item \textbf{Empirical Subject:} The academic retention, course completion, and graduation outcomes of undergraduate students in administrative higher education registries, specifically evaluated on the UCI ID 697 dataset ($N = 4,424$ institutional records from the Polytechnic Institute of Portalegre) and the large-scale Kaggle Playground Series 4 Episode 6 benchmark ($N = 76,518$ records).
\end{enumerate}

\subsection{Research Scope and Boundaries}
The empirical and technical boundaries of this investigation are defined as follows:
\begin{itemize}
    \item \textbf{Institutional Scope:} Undergraduate degree programs across 17 fields of study, tracking students from initial enrollment through terminal graduation or dropout.
    \item \textbf{Observational Policy Scope:} Analysis of state-subsidized operational scholarships ($T$) administered by national and university social welfare services, evaluated after controlling for all observable pre-treatment covariates.
    \item \textbf{Methodological Boundaries:} We focus on DAG-based causal identification under the unconfoundedness assumption ($(Y(1), Y(0)) \perp\!\!\!\perp T \mid \mathbf{Z}$), accompanied by formal sensitivity bounds for potential unmeasured confounding. Network interference and general equilibrium labor market dynamics are outside the current monograph scope.
\end{itemize}
"""
    with open(base_dir / "1_4_Research_subjects_and_scope.tex", "w", encoding="utf-8") as f:
        f.write(c1_4)

    # 1_5
    c1_5 = r"""\section{Contributions of the Study}

This investigation delivers six key methodological, computational, and policy contributions:

\begin{enumerate}
    \item \textbf{Controlled $2 \times 2$ Factorial Benchmarking:} We design and execute an end-to-end framework systematically crossing graph topologies (Domain DAG vs. Learned DAG) and parameterizations (Tabular CPT vs. Neural CPD), isolating the specific drivers of predictive and causal performance across 5-fold cross-validation.
    \item \textbf{Resolution of Continuous Structure Discovery via Temporal Masking:} We show that unconstrained continuous discovery (NOTEARS) suffers from variance sortability failure (inverting temporal causality $Y \to T$). We formulate an exact Temporal Forbidden Mask matrix that restores valid causal directionality.
    \item \textbf{Vectorized $\mathcal{O}(1)$ Counterfactual Tensor Query Engine:} We develop a PyTorch-based tensor engine that executes counterfactual interventional queries across entire cohorts in milliseconds---achieving a $>3,000\times$ speedup over iterative variable elimination.
    \item \textbf{Quantification of Confounding Bias and Causal Invariance:} We demonstrate that $31.86\%$ of the apparent raw scholarship graduation advantage is spurious selection bias (naive gap $33.9\%$ vs. true causal $ATE = +23.09\%$). Furthermore, neural CPD parameterization preserves exact causal invariance while improving probability calibration.
    \item \textbf{Causal Mediation and Subgroup Effect Heterogeneity:} We prove that $21.53\%$ of the scholarship's effect operates through early academic momentum ($NIE = +4.65\%$), while $78.47\%$ acts as direct financial protection ($NDE = +16.95\%$). Furthermore, non-traditional older students ($>20$ years) and indebted students exhibit treatment multipliers of $1.72\times$ and $2.03\times$.
    \item \textbf{Zero-Cost Policy Optimization and Out-of-Distribution Invariance:} Reallocating the historical quota of 969 scholarships via CATE ranking generates $\mathbf{+164}$ additional graduates annually ($+4.52\%$ graduation surge) at zero added fiscal cost. Finally, we demonstrate zero-shot causal transfer invariance ($|\Delta ATE| = 0.0010$) across 76,518 instances from Kaggle S4E6.
\end{enumerate}
"""
    with open(base_dir / "1_5_Contributions_of_the_study.tex", "w", encoding="utf-8") as f:
        f.write(c1_5)

    # 1_6
    c1_6 = r"""\section{Report Structure}

The remainder of this research monograph is organized as follows:
\begin{itemize}
    \item \textbf{Chapter 2 (Theoretical Foundations and Literature Review):} Reviews Probabilistic Graphical Models, Judea Pearl's structural causal hierarchy, continuous structure discovery (NOTEARS), semi-parametric causal estimators (AIPW, Causal Forests), Cinelli-Hazlett sensitivity algebra, and educational data mining literature.
    \item \textbf{Chapter 3 (Methodology and System Architecture):} Formalizes the 7-node Primary Domain DAG, specifies the $2 \times 2$ factorial model space, details the Deep Neural CPD architecture and Temporal Forbidden Mask, and derives the vectorized tensor interventional query engine.
    \item \textbf{Chapter 4 (Empirical Evaluation and Core Benchmark Results):} Presents Table A (held-out predictive performance), Table B (causal interventions and ATE), and Table C (adversarial DoWhy refutations and runtime), followed by diagnostic ablations on regularization and temporal masking.
    \item \textbf{Chapter 5 (Advanced Causal Extensions and Policy Optimization):} Explores continuous GPA and retention estimands, causal mediation path decompositions, heterogeneous treatment effects (CATE), Cinelli-Hazlett sensitivity contours, budget-constrained policy optimization, and large-scale Kaggle S4E6 transferability.
    \item \textbf{Chapter 6 (Conclusions, Limitations, and Future Directions):} Synthesizes the core theoretical and empirical findings, outlines actionable institutional policy directives, assesses threats to validity, and identifies promising frontiers for causal educational informatics.
\end{itemize}
"""
    with open(base_dir / "1_6_Report_structure.tex", "w", encoding="utf-8") as f:
        f.write(c1_6)

    print("Successfully condensed all Chapter 1 files!")

if __name__ == '__main__':
    update_c1()
