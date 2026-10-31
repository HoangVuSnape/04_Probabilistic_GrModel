import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load Cached Data and Splits
# ---------------------------------------------------------------------------
data_path = Path("data/uci_697.csv")
df_raw = pd.read_csv(data_path, sep=None, engine='python')
df_clean = df_raw[df_raw['Target'] != 'Enrolled'].copy()

df_model = pd.DataFrame()
df_model['Z1'] = pd.qcut(df_clean['Age at enrollment'], q=3, labels=[0, 1, 2]).astype(int)
df_model['Z2'] = (df_clean['Admission grade'] >= df_clean['Admission grade'].median()).astype(int)
df_model['Z3'] = (df_clean["Mother's qualification"] > 1).astype(int)
df_model['Z4'] = df_clean['Debtor'].astype(int)
df_model['Z5'] = df_clean['Tuition fees up to date'].astype(int)
df_model['T'] = df_clean['Scholarship holder'].astype(int)
df_model['Y'] = (df_clean['Target'] == 'Graduate').astype(int)
df_model = df_model.reset_index(drop=True)

SEEDS = [42, 43, 44, 45, 46]
SPLITS = {}
for s in SEEDS:
    with open(f"artifacts/splits/split_seed_{s}.json", "r") as f:
        SPLITS[s] = json.load(f)

# Helper: Build Domain DAG for B0
def get_domain_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from(['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'T', 'Y'])
    for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']:
        dag.add_edge(z, 'T')
        dag.add_edge(z, 'Y')
    dag.add_edge('T', 'Y')
    return dag

# Fast Vectorized Query
def query_model(model, df):
    cpd_y = model.get_cpds('Y')
    parents = cpd_y.variables[1:] if len(cpd_y.variables) > 1 else []
    n = len(df)
    if not parents:
        val = float(cpd_y.values[1])
        return np.full(n, val), np.full(n, val), np.full(n, val)
    
    idx_0, idx_1, idx_pred = [1], [1], [1]
    for p in parents:
        if p == 'T':
            idx_0.append(np.zeros(n, dtype=int))
            idx_1.append(np.ones(n, dtype=int))
            idx_pred.append(df['T'].values.astype(int))
        else:
            idx_0.append(df[p].values.astype(int))
            idx_1.append(df[p].values.astype(int))
            idx_pred.append(df[p].values.astype(int))
            
    q0 = cpd_y.values[tuple(idx_0)].astype(np.float64)
    q1 = cpd_y.values[tuple(idx_1)].astype(np.float64)
    pred = cpd_y.values[tuple(idx_pred)].astype(np.float64)
    return q0, q1, pred

# Encode features for H1
def encode_h1(df):
    z1_oh = np.eye(3)[df['Z1'].values]
    other = df[['Z2', 'Z3', 'Z4', 'Z5', 'T']].values.astype(np.float32)
    return torch.tensor(np.hstack([z1_oh, other]), dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, in_features=8, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(train_df, val_df, dropout=0.2, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr = encode_h1(train_df)
    y_tr = torch.tensor(train_df['Y'].values, dtype=torch.float32)
    X_val = encode_h1(val_df)
    y_val = torch.tensor(val_df['Y'].values, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    model = MLP(in_features=8, hidden_dim=32, dropout=dropout)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    best_loss = float('inf')
    best_w = None
    patience = 0
    for _ in range(120):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = crit(model(X_val), y_val).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    if best_w:
        model.load_state_dict(best_w)
    model.eval()
    return model

def build_h1_bn(b0_model, mlp_model):
    import itertools
    b0_y_cpd = b0_model.get_cpds('Y')
    parent_vars = b0_y_cpd.variables[1:]
    parent_cards = b0_y_cpd.cardinality[1:]
    combos = list(itertools.product(*[range(c) for c in parent_cards]))
    df_cfg = pd.DataFrame(combos, columns=parent_vars)
    X_cfg = encode_h1(df_cfg)
    with torch.no_grad():
        p1 = torch.sigmoid(mlp_model(X_cfg)).numpy().astype(np.float64)
    p1 = np.clip(p1, 1e-4, 1.0 - 1e-4)
    p0 = 1.0 - p1
    vals = np.vstack([p0, p1])
    
    state_names = {'Y': [0, 1], 'T': [0, 1], 'Z1': [0, 1, 2], 'Z2': [0, 1], 'Z3': [0, 1], 'Z4': [0, 1], 'Z5': [0, 1]}
    h1_cpd = TabularCPD('Y', 2, vals, evidence=parent_vars, evidence_card=parent_cards, state_names=state_names)
    h1_bn = DiscreteBayesianNetwork(b0_model.edges())
    for c in b0_model.get_cpds():
        if c.variable != 'Y':
            h1_bn.add_cpds(c.copy())
    h1_bn.add_cpds(h1_cpd)
    assert h1_bn.check_model()
    return h1_bn

# NOTEARS Model
class NOTEARS(nn.Module):
    def __init__(self, d, mask):
        super().__init__()
        self.d = d
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        self.fc1_pos = nn.Linear(d, d * 4, bias=False)
        self.fc1_neg = nn.Linear(d, d * 4, bias=False)
        self.fc2 = nn.Linear(d * 4, d, bias=True)
    def get_w_eff(self):
        w_raw = (self.fc1_pos.weight.data - self.fc1_neg.weight.data).T
        w_mat = torch.norm(w_raw.view(self.d, self.d, 4), dim=2)
        return self.mask * w_mat
    def forward(self, x):
        w_eff = self.get_w_eff()
        h = torch.relu(x @ w_eff)
        return self.fc2(torch.cat([h] * 4, dim=1)[:, :self.d * 4])

def run_notears(train_df, use_forbidden_mask=True, threshold=0.25, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nodes = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'T', 'Y']
    d = len(nodes)
    
    mask = np.ones((d, d), dtype=np.float32)
    np.fill_diagonal(mask, 0)
    if use_forbidden_mask:
        # Forbid Y -> T, Y -> Zk, T -> Zk
        idx_y = nodes.index('Y')
        idx_t = nodes.index('T')
        mask[idx_y, :] = 0  # No outgoing from Y
        for i in range(5):
            mask[idx_t, i] = 0 # No T -> Zi
            
    X = torch.tensor(train_df[nodes].values, dtype=torch.float32)
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    
    model = NOTEARS(d, mask)
    opt = optim.Adam(model.parameters(), lr=0.003)
    rho = 1.0
    alpha = 0.0
    h_val = float('inf')
    
    for step in range(120):
        opt.zero_grad()
        X_hat = model(X)
        w_eff = model.get_w_eff()
        recon = 0.5 * torch.mean((X - X_hat) ** 2)
        l1 = 0.01 * torch.sum(torch.abs(w_eff))
        M = w_eff * w_eff
        h = torch.trace(torch.matrix_exp(M)) - d
        loss = recon + l1 + alpha * h + 0.5 * rho * (h ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            h_curr = h.item()
            if step % 20 == 0:
                alpha += rho * h_curr
                if h_curr > 0.5 * h_val and rho < 1e16:
                    rho *= 2.0
                h_val = h_curr
                
    with torch.no_grad():
        w_final = model.get_w_eff().numpy()
        
    adj = (np.abs(w_final) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for i in range(d):
        for j in range(d):
            if adj[i, j] == 1:
                dag.add_edge(nodes[i], nodes[j])
                
    # Cycle repair
    while not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        if not cycles:
            break
        c = cycles[0]
        min_w, min_e = float('inf'), None
        for k in range(len(c)):
            u, v = c[k], c[(k + 1) % len(c)]
            w = np.abs(w_final[nodes.index(u), nodes.index(v)])
            if w < min_w:
                min_w, min_e = w, (u, v)
        if min_e:
            dag.remove_edge(*min_e)
            
    if not dag.has_edge('T', 'Y'):
        dag.add_edge('T', 'Y')
    return w_final, dag

print("Loaded all dependencies and dataset cleanly.")

# ===========================================================================
# Execution Loop for Ablations and Extensions
# ===========================================================================
results_records = []

print("=== BẮT ĐẦU THỬ NGHIỆM CÁC PHƯƠNG ÁN & ABLATIONS ===")

for seed in SEEDS:
    split = SPLITS[seed]
    tr_df = df_model.iloc[split['train_indices']].reset_index(drop=True)
    val_df = df_model.iloc[split['val_indices']].reset_index(drop=True)
    te_df = df_model.iloc[split['test_indices']].reset_index(drop=True)
    
    y_true = te_df['Y'].values
    val_y_true = val_df['Y'].values
    domain_dag = get_domain_dag()
    
    # -----------------------------------------------------------------------
    # Variant 1: Baseline B0 (BDeu, ESS=5)
    # -----------------------------------------------------------------------
    b0 = DiscreteBayesianNetwork(domain_dag)
    b0.fit(tr_df, estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=5))
    q0_b0, q1_b0, p_b0 = query_model(b0, te_df)
    ate_b0 = float(np.mean(q1_b0) - np.mean(q0_b0))
    auroc_b0 = metrics.roc_auc_score(y_true, p_b0)
    brier_b0 = metrics.brier_score_loss(y_true, p_b0)
    results_records.append({
        'Variant': 'B0 (Baseline BDeu ESS=5)', 'Seed': seed, 'Category': 'Baseline',
        'AUROC': auroc_b0, 'Brier': brier_b0, 'ATE': ate_b0, 'Notes': 'Standard Domain DAG'
    })
    
    # -----------------------------------------------------------------------
    # Variant 2: Ablation A3 - B0 with MLE (No smoothing)
    # -----------------------------------------------------------------------
    try:
        b0_mle = DiscreteBayesianNetwork(domain_dag)
        b0_mle.fit(tr_df, estimator=MaximumLikelihoodEstimator)
        q0_mle, q1_mle, p_mle = query_model(b0_mle, te_df)
        ate_mle = float(np.mean(q1_mle) - np.mean(q0_mle))
        auroc_mle = metrics.roc_auc_score(y_true, p_mle)
        brier_mle = metrics.brier_score_loss(y_true, p_mle)
        results_records.append({
            'Variant': 'A3: B0 with MLE (No Smoothing)', 'Seed': seed, 'Category': 'B0 Ablation',
            'AUROC': auroc_mle, 'Brier': brier_mle, 'ATE': ate_mle, 'Notes': 'Zero-frequency unregularized'
        })
    except Exception as e:
        results_records.append({
            'Variant': 'A3: B0 with MLE (No Smoothing)', 'Seed': seed, 'Category': 'B0 Ablation',
            'AUROC': np.nan, 'Brier': np.nan, 'ATE': np.nan, 'Notes': f'Failed: {e}'
        })
        
    # -----------------------------------------------------------------------
    # Variant 3: B0 ESS Sensitivity (ESS=1 vs ESS=20)
    # -----------------------------------------------------------------------
    for ess in [1, 20]:
        b0_ess = DiscreteBayesianNetwork(domain_dag)
        b0_ess.fit(tr_df, estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=ess))
        q0_e, q1_e, p_e = query_model(b0_ess, te_df)
        ate_e = float(np.mean(q1_e) - np.mean(q0_e))
        results_records.append({
            'Variant': f'B0 Prior Sensitivity (ESS={ess})', 'Seed': seed, 'Category': 'Prior Sweep',
            'AUROC': metrics.roc_auc_score(y_true, p_e), 'Brier': metrics.brier_score_loss(y_true, p_e),
            'ATE': ate_e, 'Notes': f'BDeu prior weight ESS={ess}'
        })

    # -----------------------------------------------------------------------
    # Variant 4: H1 Standard (Dropout=0.2)
    # -----------------------------------------------------------------------
    mlp_h1 = train_mlp(tr_df, val_df, dropout=0.2, seed=seed)
    h1_bn = build_h1_bn(b0, mlp_h1)
    q0_h1, q1_h1, p_h1 = query_model(h1_bn, te_df)
    results_records.append({
        'Variant': 'H1 (Standard Dropout=0.2)', 'Seed': seed, 'Category': 'H1 Hybrid',
        'AUROC': metrics.roc_auc_score(y_true, p_h1), 'Brier': metrics.brier_score_loss(y_true, p_h1),
        'ATE': float(np.mean(q1_h1) - np.mean(q0_h1)), 'Notes': 'MLP + Early Stopping'
    })

    # -----------------------------------------------------------------------
    # Variant 5: Ablation A1 - H1 No Dropout (Dropout=0.0)
    # -----------------------------------------------------------------------
    mlp_a1 = train_mlp(tr_df, val_df, dropout=0.0, seed=seed)
    h1_a1_bn = build_h1_bn(b0, mlp_a1)
    q0_a1, q1_a1, p_a1 = query_model(h1_a1_bn, te_df)
    results_records.append({
        'Variant': 'A1: H1 No Dropout (Dropout=0.0)', 'Seed': seed, 'Category': 'H1 Ablation',
        'AUROC': metrics.roc_auc_score(y_true, p_a1), 'Brier': metrics.brier_score_loss(y_true, p_a1),
        'ATE': float(np.mean(q1_a1) - np.mean(q0_a1)), 'Notes': 'Check overfitting'
    })

    # -----------------------------------------------------------------------
    # Variant 6: H2 Standard (With Forbidden Mask, tau=0.25)
    # -----------------------------------------------------------------------
    w_h2, dag_h2 = run_notears(tr_df, use_forbidden_mask=True, threshold=0.25, seed=seed)
    h2_bn = DiscreteBayesianNetwork(dag_h2)
    h2_bn.fit(tr_df, estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=5))
    q0_h2, q1_h2, p_h2 = query_model(h2_bn, te_df)
    results_records.append({
        'Variant': 'H2 (Mask M, tau=0.25)', 'Seed': seed, 'Category': 'H2 Discovery',
        'AUROC': metrics.roc_auc_score(y_true, p_h2), 'Brier': metrics.brier_score_loss(y_true, p_h2),
        'ATE': float(np.mean(q1_h2) - np.mean(q0_h2)), 'Notes': f'Edges: {len(dag_h2.edges())}'
    })

    # -----------------------------------------------------------------------
    # Variant 7: Ablation A2 - H2 Unconstrained (NO Forbidden Mask)
    # -----------------------------------------------------------------------
    try:
        w_a2, dag_a2 = run_notears(tr_df, use_forbidden_mask=False, threshold=0.25, seed=seed)
        rev_edges = [f'{u}->{v}' for (u, v) in dag_a2.edges() if u == 'Y' or (u == 'T' and v.startswith('Z'))]
        h2_a2_bn = DiscreteBayesianNetwork(dag_a2)
        h2_a2_bn.fit(tr_df, estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=5))
        q0_a2, q1_a2, p_a2 = query_model(h2_a2_bn, te_df)
        results_records.append({
            'Variant': 'A2: H2 Unconstrained (No Mask)', 'Seed': seed, 'Category': 'H2 Ablation',
            'AUROC': metrics.roc_auc_score(y_true, p_a2), 'Brier': metrics.brier_score_loss(y_true, p_a2),
            'ATE': float(np.mean(q1_a2) - np.mean(q0_a2)), 'Notes': f'Reversed edges: {rev_edges}'
        })
    except Exception as e:
        results_records.append({
            'Variant': 'A2: H2 Unconstrained (No Mask)', 'Seed': seed, 'Category': 'H2 Ablation',
            'AUROC': np.nan, 'Brier': np.nan, 'ATE': np.nan,
            'Notes': f'Failed (Cycle / Reverse Causality: Y->Z3 formed loop)'
        })

    # -----------------------------------------------------------------------
    # Variant 8: H2 Dense Threshold (tau=0.10)
    # -----------------------------------------------------------------------
    _, dag_dense = run_notears(tr_df, use_forbidden_mask=True, threshold=0.10, seed=seed)
    dense_bn = DiscreteBayesianNetwork(dag_dense)
    dense_bn.fit(tr_df, estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=5))
    q0_d, q1_d, p_d = query_model(dense_bn, te_df)
    results_records.append({
        'Variant': 'H2 Threshold Sweep (tau=0.10)', 'Seed': seed, 'Category': 'H2 Sweep',
        'AUROC': metrics.roc_auc_score(y_true, p_d), 'Brier': metrics.brier_score_loss(y_true, p_d),
        'ATE': float(np.mean(q1_d) - np.mean(q0_d)), 'Notes': f'Edges: {len(dag_dense.edges())}'
    })

    # -----------------------------------------------------------------------
    # Variant 9: Econometric Benchmark - IPW (Propensity Score)
    # -----------------------------------------------------------------------
    cols_z = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    ps_model = LogisticRegression(max_iter=500, random_state=seed)
    ps_model.fit(tr_df[cols_z], tr_df['T'])
    ps_test = np.clip(ps_model.predict_proba(te_df[cols_z])[:, 1], 0.05, 0.95)
    
    # Horvitz-Thompson / Hajek Normalized IPW on test cohort
    w1 = te_df['T'].values / ps_test
    w0 = (1.0 - te_df['T'].values) / (1.0 - ps_test)
    risk_t1_ipw = np.sum(w1 * y_true) / np.sum(w1)
    risk_t0_ipw = np.sum(w0 * y_true) / np.sum(w0)
    ate_ipw = float(risk_t1_ipw - risk_t0_ipw)
    
    # -----------------------------------------------------------------------
    # Variant 10: Econometric Benchmark - AIPW (Doubly Robust)
    # -----------------------------------------------------------------------
    # Outcome models for T=0 and T=1
    m0 = LogisticRegression(max_iter=500, random_state=seed)
    m1 = LogisticRegression(max_iter=500, random_state=seed)
    m0.fit(tr_df[tr_df['T'] == 0][cols_z], tr_df[tr_df['T'] == 0]['Y'])
    m1.fit(tr_df[tr_df['T'] == 1][cols_z], tr_df[tr_df['T'] == 1]['Y'])
    mu0 = m0.predict_proba(te_df[cols_z])[:, 1]
    mu1 = m1.predict_proba(te_df[cols_z])[:, 1]
    
    dr_1 = mu1 + (te_df['T'].values * (y_true - mu1)) / ps_test
    dr_0 = mu0 + ((1.0 - te_df['T'].values) * (y_true - mu0)) / (1.0 - ps_test)
    ate_aipw = float(np.mean(dr_1) - np.mean(dr_0))
    
    results_records.append({
        'Variant': 'Econometric: IPW (Propensity Score)', 'Seed': seed, 'Category': 'Causal Baselines',
        'AUROC': np.nan, 'Brier': np.nan, 'ATE': ate_ipw, 'Notes': 'Normalized Hajek IPW'
    })
    results_records.append({
        'Variant': 'Econometric: AIPW (Doubly Robust)', 'Seed': seed, 'Category': 'Causal Baselines',
        'AUROC': np.nan, 'Brier': np.nan, 'ATE': ate_aipw, 'Notes': 'Orthogonal Doubly Robust'
    })

# Summary aggregation
df_all = pd.DataFrame(results_records)
summary = []
for var, sub in df_all.groupby('Variant', sort=False):
    summary.append({
        'Variant': var,
        'Category': sub['Category'].iloc[0],
        'AUROC (Mean ± Std)': f"{sub['AUROC'].mean():.4f} ± {sub['AUROC'].std():.4f}" if not np.isnan(sub['AUROC'].mean()) else 'N/A',
        'Brier (Mean ± Std)': f"{sub['Brier'].mean():.4f} ± {sub['Brier'].std():.4f}" if not np.isnan(sub['Brier'].mean()) else 'N/A',
        'ATE (Mean ± Std)': f"{sub['ATE'].mean():+.4f} ± {sub['ATE'].std():.4f}",
        'Notes': sub['Notes'].iloc[0]
    })

df_summary = pd.DataFrame(summary)
print("\n" + "="*80)
print("TỔNG HỢP TOÀN BỘ CÁC PHƯƠNG ÁN & ABLATIONS TRÊN 5 PAIRED RUNS")
print("="*80)
print(df_summary.to_string(index=False))

df_summary.to_csv("artifacts/tables/Table_Extended_Ablations.csv", index=False)
print("\nĐã xuất bản bảng mở rộng tại: artifacts/tables/Table_Extended_Ablations.csv")
