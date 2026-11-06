from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\2_Background")

def update_c2():
    # 2_1
    c2_1 = r"""\section{Probabilistic Graphical Models and Bayesian Networks}
\label{sec:bg_pgm}

Probabilistic Graphical Models (PGMs) encode joint probability distributions over multivariate random vectors by exploiting conditional independence assertions represented in directed graphs \parencite{pearl2009causality, ankan2015pgmpy}. Let $\mathbf{X} = \{X_1, \dots, X_d\}$ denote a set of random variables. A Bayesian Network (BN) is defined as a pair $\mathcal{B} = (\mathcal{G}, \mathbf{P})$, where $\mathcal{G} = (\mathbf{V}, \mathbf{E})$ is a Directed Acyclic Graph (DAG) and $\mathbf{P} = \{P(X_i \mid \text{Pa}_{\mathcal{G}}(X_i))\}_{i=1}^d$ is a collection of Conditional Probability Distributions (CPDs) conditioned on the parental set $\text{Pa}_{\mathcal{G}}(X_i) = \{X_j \in \mathbf{V} : (X_j \to X_i) \in \mathbf{E}\}$.

\subsection{Factorization and d-Separation}

Under the Local Markov Assumption, each variable $X_i$ is conditionally independent of its non-descendants given its direct parents: $X_i \perp\!\!\!\perp \text{NonDescendants}(X_i) \mid \text{Pa}(X_i)$. Consequently, the joint distribution factorizes recursively as:
\begin{equation}
    P(X_1, \dots, X_d) = \prod_{i=1}^d P(X_i \mid \text{Pa}_{\mathcal{G}}(X_i)).
    \label{eq:bn_factorization}
\end{equation}
Equation \eqref{eq:bn_factorization} reduces an unconstrained joint probability mass function of size $\mathcal{O}(2^d)$ to $\mathcal{O}(d \cdot 2^k)$, where $k = \max_i |\text{Pa}(X_i)| \ll d$.

Conditional independencies in $\mathcal{G}$ are evaluated graphically via \textit{directed separation} (d-separation) \parencite{pearl2009causality}:
\begin{definition}[d-separation]
A trail $p$ between $V_1$ and $V_m$ is d-separated (blocked) by conditioning set $\mathbf{Z}$ if: (1) $p$ contains a chain $V_{k-1} \to V_k \to V_{k+1}$ or fork $V_{k-1} \leftarrow V_k \to V_{k+1}$ with $V_k \in \mathbf{Z}$; or (2) $p$ contains a collider $V_{k-1} \to V_k \leftarrow V_{k+1}$ where neither $V_k$ nor any descendant belongs to $\mathbf{Z}$. If all trails between disjoint sets $\mathbf{X}$ and $\mathbf{Y}$ are blocked by $\mathbf{Z}$, then $(\mathbf{X} \perp_{\mathcal{G}} \mathbf{Y} \mid \mathbf{Z}) \implies \mathbf{X} \perp\!\!\!\perp \mathbf{Y} \mid \mathbf{Z}$.
\end{definition}

\subsection{The Parameter Explosion Dilemma in Discrete CPTs}

In standard discrete BNs, tabular Conditional Probability Tables (CPTs) parameterize $P(Y \mid \text{Pa}(Y))$. For discrete target $Y$ with support $|\mathcal{Y}|$ and parents $\{Z_1, \dots, Z_m\}$, the number of free parameters scales exponentially:
\begin{equation}
    K_{\text{CPT}} = (|\mathcal{Y}| - 1) \prod_{j=1}^m |\mathcal{Z}_j|.
    \label{eq:cpt_explosion}
\end{equation}
In higher education registries containing multiple continuous demographic variables, continuous features must be coarsely discretized (e.g., binning grades into 3--5 quantiles). When $m = 5$ parents are discretized into 5 bins each, a single binary node requires estimating $(2 - 1) \cdot 5^5 = 3,125$ conditional probability parameters. In realistic institutional cohorts ($N \approx 3,000 \sim 4,000$), many conditioning cells contain zero observations, causing Maximum Likelihood Estimation (MLE) to produce degenerate zero-frequency predictions ($0/0$) or fall back on uninformative uniform priors, degrading held-out calibration.
"""
    with open(base_dir / "2_1_Probabilistic_Graphical_Models.tex", "w", encoding="utf-8") as f:
        f.write(c2_1)

    # 2_2
    c2_2 = r"""\section{Structural Causal Models and Pearl's Causal Hierarchy}
\label{sec:bg_scm}

While observational Bayesian networks model $P(\mathbf{X})$, distinct DAG structures may belong to the same Markov Equivalence Class yet prescribe contradictory policy interventions \parencite{pearl2009causality}. Judea Pearl formulated Structural Causal Models (SCMs) and established the Three-Layer Causal Hierarchy (Table \ref{tab:causal_hierarchy}).

\begin{table}[htbp]
    \centering
    \small
    \caption{Pearl's Three-Layer Causal Hierarchy and Educational Policy Equivalents.}
    \label{tab:causal_hierarchy}
    \begin{tabularx}{\textwidth}{p{2.2cm}p{3.2cm}p{4.2cm}X}
        \toprule
        \textbf{Layer} & \textbf{Mathematical Form} & \textbf{Typical Query} & \textbf{Educational Domain Meaning} \\
        \midrule
        \textbf{$L_1$: Association} & $P(Y \mid T=t)$ & What does observing a scholarship tell us about graduation? & Observational correlation; confounded by socioeconomic privilege. \\
        \textbf{$L_2$: Intervention} & $P(Y \mid do(T=t))$ & What will happen if the university mandates aid for this group? & Institutional policy effect; graph surgery cuts incoming confounding arrows. \\
        \textbf{$L_3$: Counterfactuals} & $P(Y_{T=1} \mid T=0, Y=0)$ & Would a student who dropped out have graduated had aid been given? & Individual retrospective accountability and algorithmic fairness. \\
        \bottomrule
    \end{tabularx}
\end{table}

\subsection{Structural Assignments and Graph Surgery}

An SCM is a 4-tuple $\mathcal{M} = \langle \mathbf{U}, \mathbf{V}, \mathbf{F}, P(\mathbf{U}) \rangle$, where $\mathbf{U}$ are exogenous background variables governed by $P(\mathbf{U})$, $\mathbf{V}$ are endogenous variables, and $\mathbf{F} = \{f_i\}_{i=1}^d$ are deterministic structural assignments:
\begin{equation}
    V_i \leftarrow f_i(\text{Pa}_{\mathcal{G}}(V_i), U_i), \quad \forall i \in \{1, \dots, d\}.
    \label{eq:scm_assignment}
\end{equation}
An external physical intervention setting variable $T$ to constant $t$, denoted $do(T=t)$, replaces structural equation $f_T$ with the assignment $T \leftarrow t$. Graphically, this corresponds to \textit{graph surgery}: all directed edges entering $T$ from its parents $\text{Pa}(T)$ are removed in the sub-graph $\mathcal{G}_{\overline{T}}$, severing all incoming confounding paths.

\subsection{The Backdoor Criterion and Identification Formula}

\begin{definition}[Backdoor Criterion \parencite{pearl2009causality}]
A set of variables $\mathbf{Z}$ satisfies the Backdoor Criterion relative to an ordered pair of variables $(T, Y)$ in a DAG $\mathcal{G}$ if:
\begin{enumerate}
    \item No node in $\mathbf{Z}$ is a causal descendant of treatment $T$ (i.e., $\mathbf{Z} \cap \text{Descendants}(T) = \emptyset$).
    \item $\mathbf{Z}$ blocks every backdoor path between $T$ and $Y$ that enters $T$ through an incoming arrow in $\mathcal{G}_{\underline{T}}$.
\end{enumerate}
\end{definition}

When backdoor admissibility holds, the interventional distribution is non-parametrically identified by the Backdoor Adjustment Formula:
\begin{equation}
    P(Y = y \mid do(T = t)) = \sum_{\mathbf{z} \in \mathcal{Z}} P(Y = y \mid T = t, \mathbf{Z} = \mathbf{z}) P(\mathbf{Z} = \mathbf{z}).
    \label{eq:backdoor_adjustment}
\end{equation}
Correspondingly, the Average Treatment Effect ($ATE$) of shifting treatment from control ($T=0$) to intervention ($T=1$) on binary outcome $Y \in \{0, 1\}$ is:
\begin{equation}
\begin{aligned}
    ATE &= \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)] \\
        &= \sum_{\mathbf{z}} \left[ P(Y=1 \mid T=1, \mathbf{z}) - P(Y=1 \mid T=0, \mathbf{z}) \right] P(\mathbf{z}).
\end{aligned}
\label{eq:ate_definition}
\end{equation}
"""
    with open(base_dir / "2_2_Structural_Causal_Models.tex", "w", encoding="utf-8") as f:
        f.write(c2_2)

    # 2_3
    c2_3 = r"""\section{Causal Discovery: From Combinatorial Search to Continuous Optimization}
\label{sec:bg_causal_discovery}

When the true data-generating causal DAG $\mathcal{G}^*$ is unknown a priori, causal discovery algorithms uncover directed graph structures directly from observational samples $\mathcal{D} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$ \parencite{zheng2018notears}.

\subsection{Continuous Optimization via NOTEARS}

Combinatorial score-based methods (e.g., GES, Hill-Climbing) navigate a discrete search space that scales super-exponentially with dimension $d$: $a_d = \sum_{k=1}^d (-1)^{k-1} \binom{d}{k} 2^{k(d-k)} a_{d-k}$ ($a_{10} \approx 4.2 \times 10^{18}$). \textcite{zheng2018notears} fundamentally transformed this problem into continuous constrained optimization by establishing an exact, differentiable algebraic characterization of acyclicity:
\begin{theorem}[NOTEARS Acyclicity Constraint \parencite{zheng2018notears}]
A real-valued adjacency matrix $W \in \mathbb{R}^{d \times d}$ corresponds to a Directed Acyclic Graph if and only if:
\begin{equation}
    h(W) \equiv \operatorname{tr}\left( e^{W \circ W} \right) - d = 0,
    \label{eq:notears_constraint}
\end{equation}
where $\circ$ denotes Hadamard element-wise multiplication, $e^M = \sum_{k=0}^\infty \frac{M^k}{k!}$ is the matrix exponential, and $\operatorname{tr}(\cdot)$ is matrix trace.
\end{theorem}
The continuous structure learning objective combines least-squares loss, $\ell_1$-sparsity regularization, and the acyclicity penalty:
\begin{equation}
    \min_{W \in \mathbb{R}^{d \times d}} \quad \frac{1}{2N} \|X - XW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t.} \quad h(W) = 0,
    \label{eq:notears_objective}
\end{equation}
solved via the Augmented Lagrangian method with subproblem updates: $\mathcal{L}_c(W, \alpha) = F(W) + \alpha h(W) + \frac{c}{2} h(W)^2$.

\subsection{Var-Sortability and the Temporal Forbidden Mask}

Despite mathematical elegance, continuous discovery algorithms are known to be vulnerable to \textit{variance sortability} \parencite{reisach2021varsortability}: if endogenous variables exhibit uneven empirical marginal variances, unconstrained NOTEARS systematically orients edges from high-variance variables toward low-variance variables regardless of causal ground truth. In administrative education registries where terminal dropout $Y \in \{0, 1\}$ exhibits higher variance than early entrance metrics, unconstrained NOTEARS inverts causality, discovering invalid reverse arrows $Y \to T$.

To guarantee physical causal admissibility, we enforce an exact \textbf{Temporal Forbidden Mask} $M \in \{0, 1\}^{d \times d}$, setting $M_{ij} = 0$ for any candidate edge $X_i \to X_j$ that violates physical time ordering ($t(X_i) > t(X_j)$):
\begin{equation}
    W_{\text{masked}} = W \odot M, \quad \text{where} \quad M_{ij} = \mathbb{I}(t(X_i) \le t(X_j)).
    \label{eq:temporal_mask_formula}
\end{equation}
"""
    with open(base_dir / "2_3_Causal_Discovery.tex", "w", encoding="utf-8") as f:
        f.write(c2_3)

    # 2_4
    c2_4 = r"""\section{Semi-Parametric Estimators and Sensitivity Analysis}
\label{sec:bg_estimators}

Beyond explicit graphical CPD modeling, modern causal inference relies on semi-parametric estimators operating under the Neyman-Rubin Potential Outcomes framework \parencite{rubin2005causal, hernan2020causal}. For each unit $i$, potential outcomes $Y_i(1)$ and $Y_i(0)$ represent outcomes under active intervention ($T_i=1$) and control ($T_i=0$). The observed outcome satisfies $Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$.

Identification requires two standard assumptions:
\begin{enumerate}
    \item \textbf{Conditional Exchangeability (Unconfoundedness):} $(Y(1), Y(0)) \perp\!\!\!\perp T \mid \mathbf{Z}$.
    \item \textbf{Positivity (Overlap):} $0 < \epsilon \le P(T=1 \mid \mathbf{Z}=\mathbf{z}) \le 1 - \epsilon < 1, \quad \forall \mathbf{z} \in \operatorname{supp}(\mathbf{Z})$.
\end{enumerate}

\subsection{Doubly Robust Estimation (AIPW) and Causal Forests}

The Augmented Inverse Probability Weighting (AIPW) estimator combines an outcome regression model $\mu_t(\mathbf{z}) = \mathbb{E}[Y \mid T=t, \mathbf{Z}=\mathbf{z}]$ with a propensity score model $e(\mathbf{z}) = P(T=1 \mid \mathbf{Z}=\mathbf{z})$:
\begin{equation}
    \hat{\tau}_{\text{AIPW}} = \frac{1}{N} \sum_{i=1}^N \left( \left[ \mu_1(\mathbf{z}_i) + \frac{T_i (Y_i - \mu_1(\mathbf{z}_i))}{e(\mathbf{z}_i)} \right] - \left[ \mu_0(\mathbf{z}_i) + \frac{(1 - T_i)(Y_i - \mu_0(\mathbf{z}_i))}{1 - e(\mathbf{z}_i)} \right] \right).
    \label{eq:aipw_estimator}
\end{equation}
AIPW possesses the \textit{double robustness} property: $\hat{\tau}_{\text{AIPW}}$ is consistent if \emph{either} the outcome model $\mu_t(\mathbf{z})$ \emph{or} the propensity model $e(\mathbf{z})$ is correctly specified \parencite{hernan2020causal}.

To evaluate effect heterogeneity across student subgroups, we employ Causal Forests \parencite{wager2018causalforest}, estimating the Conditional Average Treatment Effect (CATE): $\tau(\mathbf{x}) \equiv \mathbb{E}[Y(1) - Y(0) \mid \mathbf{X} = \mathbf{x}]$. Causal Forests construct adaptive nearest-neighbor kernels via orthogonal tree splitting, guaranteeing asymptotic normality and valid confidence intervals.

\subsection{Cinelli-Hazlett Sensitivity Algebra for Unmeasured Confounding}

Because unconfoundedness cannot be verified empirically from observational data, we subject causal estimates to the sensitivity algebra of \textcite{cinelli2020making}. Let $U$ denote an unobserved confounder. The omitted variable bias formula is:
\begin{equation}
    \operatorname{Bias} = |\hat{\tau} - \tau^*| = \operatorname{SE}(\hat{\tau}) \cdot \sqrt{\frac{R^2_{Y \sim U \mid T, \mathbf{Z}} \cdot R^2_{T \sim U \mid \mathbf{Z}}}{1 - R^2_{T \sim U \mid \mathbf{Z}}}} \cdot \sqrt{\text{df}}.
    \label{eq:cinelli_bias}
\end{equation}
From Equation \eqref{eq:cinelli_bias}, \textcite{cinelli2020making} define the \textit{Robustness Value} ($RV_{q=1}$), which quantifies the minimum equal explanatory power ($R^2_{Y \sim U \mid T, \mathbf{Z}} = R^2_{T \sim U \mid \mathbf{Z}} = RV$) an omitted confounder must have with both treatment and outcome to bring the estimated effect to zero.
"""
    with open(base_dir / "2_4_Statistical_Causal_Estimators.tex", "w", encoding="utf-8") as f:
        f.write(c2_4)

    # 2_5
    c2_5 = r"""\section{Related Work in Educational Data Mining and Causal Learning}
\label{sec:bg_related_work}

Higher education dropout prediction has witnessed an extensive literature within Educational Data Mining (EDM) and Learning Analytics (LA). However, prior works remain overwhelmingly confined to Layer 1 correlational prediction.

\subsection{Correlational Machine Learning for Dropout Prediction}

The benchmark Portuguese higher education dataset studied in this research was curated by \textcite{martins2021predicting} at the Polytechnic Institute of Portalegre (IPPortalegre). Their baseline evaluation benchmarked classical classifiers (MLP, Random Forests, SVM, GBDT), reporting classification accuracies between $77\%$ and $80\%$. Subsequent benchmarks on the 76,518-instance Kaggle Playground Series 4 Episode 6 dataset \parencite{kaggle2024s4e6} pushed accuracy to $83.5\%$ utilizing complex ensemble stacking. 

However, these models operate strictly as predictive black boxes:
\begin{enumerate}
    \item \textbf{Absence of Interventional Semantics:} Feature attribution metrics (SHAP, Gini importance) quantify how a feature shifts model scores, not how physical intervention alters graduation probability.
    \item \textbf{Feature Conflation and Reverse Causality:} Standard classifiers exploit post-treatment proxies (e.g., late tuition arrears) as top predictors, obscuring early actionable policy levers.
    \item \textbf{Susceptibility to Data Leakage:} Many published EDM models incorporate second-semester curricular features (e.g., \texttt{Curricular units 2nd sem grade}) to predict year-end dropout. In real university workflows, by the time semester-2 final grades appear, student departure has already occurred.
\end{enumerate}

\subsection{Causal Graphical Models in Education}

Recent studies have explored Causal Bayesian Networks (CBNs) for educational analytics \parencite{edm2024cbn}. While standard discrete CBNs satisfy Pearl's $do$-calculus, they have been constrained by discrete CPTs, requiring severe categorical binning and failing on unobserved parental combinations. Conversely, deep causal representation models (such as TARNet \parencite{shalit2017tarnet} and Dragonnet \parencite{shi2019dragonnet}) lack inspectable DAG topologies, preventing domain experts from verifying d-separation conditions and conducting transparent mediation path analysis.

\subsection{The Methodological Void and Proposed Synthesis}

This review identifies an unaddressed methodological gap:
\begin{displayquote}
\textit{There is currently no unified framework that simultaneously provides: (1) an inspectable, domain-admissible causal DAG; (2) continuous deep neural parameterization to eliminate zero-frequency CPT cell explosion; (3) an $\mathcal{O}(1)$ vectorized tensor interventional query engine; and (4) formal econometric refutation guarantees tailored for budget-constrained educational policy intervention.}
\end{displayquote}
Addressing this specific void constitutes the core technical motivation of this research.
"""
    with open(base_dir / "2_5_Related_Work_EDM.tex", "w", encoding="utf-8") as f:
        f.write(c2_5)

    print("Successfully condensed all Chapter 2 files!")

if __name__ == '__main__':
    update_c2()
