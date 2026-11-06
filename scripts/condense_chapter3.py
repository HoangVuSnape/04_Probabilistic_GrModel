from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\3_Methods_Models")

def update_c3():
    # 3_1
    c3_1 = r"""\section{The Primary Domain Causal DAG and Identification}
\label{sec:meth_domain_dag}

The structural foundation of our causal framework is the \textit{Primary Domain Causal DAG} ($\mathcal{G}_{\text{domain}}$), constructed in consultation with higher education pedagogical literature and institutional regulations at IPPortalegre \parencite{martins2021predicting}.

\subsection{Node Taxonomy and Operational Definitions}

The 35 administrative features are mapped into a 7-node canonical DAG:
\begin{enumerate}
    \item \textbf{Socioeconomic Profile ($Z_1$):} Student enrollment age, parental educational attainment, and housing displacement status.
    \item \textbf{Prior Academics ($Z_2$):} Secondary school exit grade and national entrance examination performance (\texttt{Admission grade}).
    \item \textbf{Financial Solvency ($Z_3$):} Baseline tuition fee arrears status upon enrollment (\texttt{Debtor}).
    \item \textbf{Primary Policy Treatment ($T$):} Binary indicator for state-subsidized operational scholarship award ($T=1$: Awarded, $T=0$: Control).
    \item \textbf{Semester 1 Credits ($M_1$):} Total ECTS course units approved in first semester (\texttt{Curricular units 1st sem approved}).
    \item \textbf{Semester 1 GPA ($M_2$):} Normalized first-semester grade point average ($[0, 20]$ scale).
    \item \textbf{Terminal Outcome ($Y$):} Definitive educational outcome ($Y=0$: Dropout, $Y=1$: Graduate).
\end{enumerate}

\subsection{Graphical Topology and Backdoor Admissibility}

Figure \ref{fig:domain_dag_tikz} displays the causal topology $\mathcal{G}_{\text{domain}}$.

\begin{figure}[htbp]
    \centering
    \resizebox{0.82\textwidth}{!}{%
    \begin{tikzpicture}[
        node distance=1.6cm and 2.2cm,
        every node/.style={draw, font=\small\sffamily, thick},
        confounder/.style={rectangle, rounded corners=3pt, fill=blue!10, text width=2.6cm, align=center},
        treatment/.style={rectangle, rounded corners=3pt, fill=red!15, draw=red!70!black, line width=1.5pt, text width=2.4cm, align=center},
        mediator/.style={rectangle, rounded corners=3pt, fill=orange!15, draw=orange!80!black, line width=1.2pt, text width=2.6cm, align=center},
        outcome/.style={rectangle, rounded corners=3pt, fill=green!20, draw=green!60!black, line width=1.5pt, text width=2.4cm, align=center},
        edge/.style={->, >=Latex, thick, draw=black!80},
        causal_edge/.style={->, >=Latex, very thick, draw=red!80!black},
        med_edge/.style={->, >=Latex, thick, draw=orange!80!black}
    ]
        % Top row: Confounders
        \node[confounder] (Z1) at (0, 2.6) {Socioeconomic ($Z_1$)};
        \node[confounder] (Z2) at (4.2, 2.6) {Prior Academics ($Z_2$)};
        \node[confounder] (Z3) at (8.4, 2.6) {Financial Solvency ($Z_3$)};
        
        % Middle row: Treatment & Mediators
        \node[treatment] (T) at (0, 0) {\textbf{Scholarship ($T$)}};
        \node[mediator] (M1) at (4.2, 0.7) {Sem 1 Credits ($M_1$)};
        \node[mediator] (M2) at (4.2, -0.7) {Sem 1 GPA ($M_2$)};
        
        % Bottom right: Outcome
        \node[outcome] (Y) at (8.4, 0) {\textbf{Graduation ($Y$)}};
        
        % Confounder Edges
        \draw[edge] (Z1) -- (T);
        \draw[edge] (Z1) -- (M1);
        \draw[edge] (Z1) to[out=-40, in=140] (Y);
        
        \draw[edge] (Z2) -- (T);
        \draw[edge] (Z2) -- (M1);
        \draw[edge] (Z2) -- (M2);
        \draw[edge] (Z2) -- (Y);
        
        \draw[edge] (Z3) -- (T);
        \draw[edge] (Z3) -- (Y);
        
        % Causal Paths
        \draw[causal_edge] (T) to[out=-30, in=-150] node[below, font=\scriptsize, draw=none, fill=none] {Direct Effect ($NDE$)} (Y);
        \draw[med_edge] (T) -- (M1);
        \draw[med_edge] (T) -- (M2);
        \draw[med_edge] (M1) -- (Y);
        \draw[med_edge] (M2) -- (Y);
        \draw[edge] (M1) -- (M2);
    \end{tikzpicture}%
    }
    \caption{The Primary Domain Causal DAG ($\mathcal{G}_{\text{domain}}$). Red node indicates treatment; orange nodes denote academic mediators; green node is terminal graduation.}
    \label{fig:domain_dag_tikz}
\end{figure}

\begin{theorem}[Backdoor Identification of Scholarship Effect]
Under $\mathcal{G}_{\text{domain}}$, the conditioning set $\mathbf{Z} = \{Z_1, Z_2, Z_3\}$ satisfies the Backdoor Criterion relative to $(T, Y)$. Consequently, $P(Y \mid do(T=t))$ is identified non-parametrically by:
\begin{equation}
    P(Y=1 \mid do(T=t)) = \sum_{z_1, z_2, z_3} P(Y=1 \mid T=t, z_1, z_2, z_3) P(z_1, z_2, z_3).
    \label{eq:backdoor_proof_domain}
\end{equation}
\end{theorem}
\begin{proof}
First, no node in $\mathbf{Z}$ is a descendant of $T$; all elements of $\mathbf{Z}$ represent pre-enrollment baseline traits determined strictly prior to scholarship allocation. Second, in the manipulated sub-graph $\mathcal{G}_{\underline{T}}$, all outgoing arrows from $T$ ($T \to M_1, T \to M_2, T \to Y$) are removed. The remaining backdoor paths connecting $T$ and $Y$ are: $T \leftarrow Z_1 \to Y$, $T \leftarrow Z_2 \to Y$, and $T \leftarrow Z_3 \to Y$. Since conditioning on $\mathbf{Z} = \{Z_1, Z_2, Z_3\}$ blocks all three forks, d-separation holds: $(T \perp_{\mathcal{G}_{\underline{T}}} Y \mid \mathbf{Z})$.
\end{proof}
"""
    with open(base_dir / "3_1_Overall_Architecture_and_Domain_DAG.tex", "w", encoding="utf-8") as f:
        f.write(c3_1)

    # 3_2
    c3_2 = r"""\section{The $2 \times 2$ Factorial Model Matrix}
\label{sec:meth_factorial_matrix}

To isolate whether empirical gains stem from graph topology versus functional parameterization, we formulate a $2 \times 2$ factorial experimental design spanning two orthogonal axes (Table \ref{tab:factorial_design}):
\begin{enumerate}
    \item \textbf{Factor 1: Graph Topology ($\mathcal{G}$):} Expert Domain DAG ($\mathcal{G}_{\text{domain}}$) vs. Data-Driven Discovered DAG ($\mathcal{G}_{\text{learned}}$ via NOTEARS with Temporal Mask).
    \item \textbf{Factor 2: Parameterization ($\mathbf{P}$):} Tabular CPTs (with BDeu Bayesian Dirichlet smoothing) vs. Continuous Neural CPDs (deep PyTorch MLPs).
\end{enumerate}

\begin{table}[htbp]
    \centering
    \small
    \caption{The $2 \times 2$ Factorial Model Architecture Space.}
    \label{tab:factorial_design}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lp{3.5cm}p{3.5cm}X}
        \toprule
        \textbf{Model Identifier} & \textbf{Graph Topology ($\mathcal{G}$)} & \textbf{CPD Parameterization ($\mathbf{P}$)} & \textbf{Methodological Role in Study} \\
        \midrule
        \textbf{$B_0$ (Baseline)} & Domain DAG ($\mathcal{G}_{\text{domain}}$) & Discrete Tabular CPT (BDeu) & Canonical discrete Bayesian Network benchmark. \\
        \textbf{$H_1$ (Neural Hybrid)} & Domain DAG ($\mathcal{G}_{\text{domain}}$) & Continuous Neural CPD (PyTorch) & Evaluates neural smoothing over expert causal structure. \\
        \textbf{$H_2$ (Learned Tabular)} & Discovered DAG ($\mathcal{G}_{\text{learned}}$) & Discrete Tabular CPT (BDeu) & Tests whether continuous structure discovery alone suffices. \\
        \textbf{$H_3$ (Fully Neural)} & Discovered DAG ($\mathcal{G}_{\text{learned}}$) & Continuous Neural CPD (PyTorch) & Full end-to-end continuous causal discovery and estimation. \\
        \bottomrule
    \end{tabular}%
    }
\end{table}

This $2 \times 2$ decomposition enables clean two-way Analysis of Variance (ANOVA): comparing $H_1$ against $B_0$ isolates the marginal effect of neural parameterization under fixed expert structure; comparing $H_2$ against $B_0$ isolates the impact of purely data-driven structure learning; and $H_3$ measures their non-linear synergy.
"""
    with open(base_dir / "3_2_Factorial_Model_Matrix.tex", "w", encoding="utf-8") as f:
        f.write(c3_2)

    # 3_3
    c3_3 = r"""\section{Neural CPD Architecture and Temporal Forbidden Masking}
\label{sec:meth_neural_cpd}

\subsection{Deep Neural Conditional Probability Architecture}

For each node $V_i \in \mathbf{V}$ possessing parents $\text{Pa}(V_i)$ in $\mathcal{G}$, we parameterize $P(V_i \mid \text{Pa}(V_i))$ via a dedicated Multi-Layer Perceptron (MLP) $f_{\theta_i}: \mathbb{R}^{|\text{Pa}(V_i)|} \to \Delta^{|\mathcal{V}_i|-1}$:
\begin{align}
    \mathbf{h}^{(1)}_i &= \operatorname{ReLU}\left( \mathbf{W}^{(1)}_i \mathbf{x}_{\text{Pa}(V_i)} + \mathbf{b}^{(1)}_i \right), \label{eq:mlp_layer1} \\
    \mathbf{h}^{(2)}_i &= \operatorname{ReLU}\left( \mathbf{W}^{(2)}_i \mathbf{h}^{(1)}_i + \mathbf{b}^{(2)}_i \right), \label{eq:mlp_layer2} \\
    \hat{\mathbf{p}}_i &= \operatorname{Softmax}\left( \mathbf{W}^{(3)}_i \mathbf{h}^{(2)}_i + \mathbf{b}^{(3)}_i \right), \label{eq:mlp_output}
\end{align}
where hidden layer dimensions are $d_1 = 64$ and $d_2 = 32$. To eliminate overfitting on finite institutional cohorts ($N \approx 3,600$), we introduce Monte Carlo Dropout ($p = 0.20$) and $\ell_2$ weight decay ($\lambda = 10^{-4}$). The optimization minimizes cross-entropy:
\begin{equation}
    \mathcal{L}(\theta_i) = -\frac{1}{N} \sum_{k=1}^N \sum_{c=1}^{|\mathcal{V}_i|} \mathbb{I}(v_{i}^{(k)} = c) \log \hat{p}_{i, c}^{(k)} + \frac{\lambda}{2} \|\theta_i\|_2^2.
    \label{eq:neural_cpd_loss}
\end{equation}

\subsection{Administrative Temporal Forbidden Mask}

Continuous structure learning via NOTEARS optimizes $\min_W \frac{1}{2N}\|X - XW\|_F^2 + \lambda \|W\|_1$ s.t. $\operatorname{tr}(e^{W \circ W}) - d = 0$. However, without temporal constraints, high-variance downstream variables invert causality (e.g., $Y \to T$). We enforce an exact \textbf{Temporal Forbidden Mask} $M \in \{0, 1\}^{7 \times 7}$ based on institutional timeline tiers:
\begin{equation}
    \text{Tier 0 (Pre-Enrollment): } \{Z_1, Z_2, Z_3\} \quad \longrightarrow \quad \text{Tier 1 (Enrollment): } \{T\} \quad \longrightarrow \quad \text{Tier 2 (Sem 1): } \{M_1, M_2\} \quad \longrightarrow \quad \text{Tier 3: } \{Y\}.
\end{equation}
Candidate adjacency weights are masked element-wise: $W_{\text{effective}} = W \odot M$, where $M_{ij} = 0$ for any reverse-time edge ($t_i > t_j$), guaranteeing 100\% temporal acyclicity.
"""
    with open(base_dir / "3_3_Neural_CPD_and_Temporal_Mask.tex", "w", encoding="utf-8") as f:
        f.write(c3_3)

    # 3_4
    c3_4 = r"""\section{Vectorized $\mathcal{O}(1)$ Tensor Query Engine}
\label{sec:meth_tensor_engine}

In standard discrete BN packages (e.g., \texttt{pgmpy}), counterfactual queries $P(Y \mid do(T=t))$ rely on exact Variable Elimination or Gibbs sampling. Variable Elimination computes intermediate factor products:
\begin{equation}
    \psi_{\text{elim}}(\mathbf{X} \setminus \{Z_j\}) = \sum_{z_j} \prod_{k \in \mathcal{F}_j} \phi_k(\text{Scope}(\phi_k)),
    \label{eq:var_elim_factor}
\end{equation}
incurring exponential complexity $\mathcal{O}(d \cdot |\mathcal{X}|^{w^*})$ in graph treewidth $w^*$. For a cohort of $N$ students, iterative elimination requires $\mathcal{O}(N \cdot d \cdot |\mathcal{X}|^{w^*})$, taking minutes per evaluation.

\subsection{Tensor Forward-Pass Intervention Formulation}

We reformulate Pearl's $do$-calculus as a single batched GPU tensor operation. Under $do(T=t)$, the treatment column is clamped across all $N$ cohort records simultaneously: $\mathbf{X}_{\text{batch}}[:, \text{idx}(T)] \leftarrow t$. For each topologically sorted child node $V_i$, conditional probabilities are evaluated via parallel tensor forward passes:
\begin{equation}
    \mathbf{P}(V_i = 1 \mid do(T=t)) = \sigma\left( f_{\theta_i}\left( \mathbf{X}_{\text{batch}}[:, \text{idx}(\text{Pa}(V_i))] \right) \right).
    \label{eq:vectorized_do_op}
\end{equation}
The Average Treatment Effect over the entire cohort is computed via a single vector reduction:
\begin{equation}
    \widehat{ATE} = \frac{1}{N} \sum_{i=1}^N \left( \mathbf{P}_i(Y = 1 \mid do(T=1)) - \mathbf{P}_i(Y = 1 \mid do(T=0)) \right).
    \label{eq:vectorized_ate_reduction}
\end{equation}
This batch tensor formulation executes in $\mathcal{O}\left( \lceil N/B \rceil \sum_l d_l d_{l-1} \right)$ operations, completing counterfactual cohort simulation in $<10$ milliseconds ($>3,000\times$ speedup over iterative variable elimination).
"""
    with open(base_dir / "3_4_Vectorized_O1_Tensor_Engine.tex", "w", encoding="utf-8") as f:
        f.write(c3_4)

    # 3_5
    c3_5 = r"""\section{DoWhy Refutation Suite and Sensitivity Algebra}
\label{sec:meth_refutations}

A causal claim is unscientific unless subjected to adversarial refutations designed to falsify its identifying assumptions \parencite{sharma2020dowhy, cinelli2020making}.

\subsection{The Three-Tier DoWhy Refutation Protocol}

We implement a three-tier adversarial refutation suite via \texttt{DoWhy} (Algorithm \ref{alg:refutation_protocol}):

\begin{algorithm}[htbp]
\caption{Three-Tier Adversarial Causal Refutation Protocol}
\label{alg:refutation_protocol}
\begin{algorithmic}[1]
\Require Model $\mathcal{M}$, Data $\mathcal{D} = \{(\mathbf{x}_i, t_i, y_i)\}_{i=1}^N$, Estimated Effect $\hat{\tau}$
\Ensure Refutation Audit Report $\mathcal{R}$
\Statex \textbf{Tier 1: Placebo Treatment Test}
\State $T_{\text{placebo}} \leftarrow \operatorname{UniformPermute}(T)$
\State $\hat{\tau}_{\text{placebo}} \leftarrow \operatorname{Estimate}(\mathcal{M}, \mathcal{D}, T_{\text{placebo}})$ \Comment{Assert $|\hat{\tau}_{\text{placebo}}| \le 0.01$, $p \ge 0.05$}
\Statex \textbf{Tier 2: Random Common Cause Test}
\State Sample synthetic noise $W_i \sim \mathcal{N}(0, 1)$, augment $\mathbf{Z}' \leftarrow \mathbf{Z} \cup \{W\}$
\State $\hat{\tau}_{\text{random}} \leftarrow \operatorname{Estimate}(\mathcal{M}, \mathcal{D}_{\text{aug}}, \mathbf{Z}')$ \Comment{Assert $|\hat{\tau} - \hat{\tau}_{\text{random}}| \le 0.01$}
\Statex \textbf{Tier 3: Data Subset Bootstrap Test}
\For{$b = 1$ to $B=100$}
    \State Sample 80\% sub-cohort $\mathcal{D}^{(b)} \subset \mathcal{D}$; compute $\hat{\tau}^{(b)} \leftarrow \operatorname{Estimate}(\mathcal{M}, \mathcal{D}^{(b)})$
\EndFor
\State $\Delta_{\text{subset}} \leftarrow \max_b \hat{\tau}^{(b)} - \min_b \hat{\tau}^{(b)}$ \Comment{Assert paired spread $\le 0.03$}
\State \Return $\mathcal{R} = \{\hat{\tau}_{\text{placebo}}, \hat{\tau}_{\text{random}}, \Delta_{\text{subset}}\}$
\end{algorithmic}
\end{algorithm}

\subsection{Cinelli-Hazlett Sensitivity Benchmarking}

To establish unobserved confounding survival thresholds, we benchmark hypothetical unmeasured confounders against observed administrative predictors:
\begin{equation}
    \operatorname{Bias}(U) = \operatorname{SE}(\hat{\tau}) \cdot \sqrt{\frac{R^2_{Y \sim U \mid T, \mathbf{Z}} \cdot R^2_{T \sim U \mid \mathbf{Z}}}{1 - R^2_{T \sim U \mid \mathbf{Z}}}} \cdot \sqrt{\text{df}}.
    \label{eq:cinelli_bias_eval}
\end{equation}
We benchmark $U$ at $1\times$, $2\times$, and $3\times$ the explanatory power of student first-semester academic performance ($Z_5$, partial $R^2 = 0.1041$), computing the Robustness Value ($RV_{q=1}$) required to overturn policy recommendations.
"""
    with open(base_dir / "3_5_Refutation_Suite_and_Sensitivity.tex", "w", encoding="utf-8") as f:
        f.write(c3_5)

    print("Successfully condensed all Chapter 3 files!")

if __name__ == '__main__':
    update_c3()
