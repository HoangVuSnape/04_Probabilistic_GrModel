# -*- coding: utf-8 -*-
"""Sinh notebook/kiem_tra_nap_lai_mo_hinh.ipynb: nạp lại B0 (JSON), H1 (.pt), H2 (JSON) từ đĩa và chấm lại trên TEST."""
import sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell(r'''# Kiểm tra nạp lại mô hình từ file đã lưu
Notebook phụ — **không huấn luyện lại**. Với mỗi seed 42–46:
* **B0**: đọc sơ đồ từ `artifacts/graphs/b0_dag_seed<s>.json`, điền bảng CPD BDeu (ESS = 5) trên TRAIN — cách điền là cố định nên phải ra đúng như cũ.
* **H1**: đọc **checkpoint `.pt`** `artifacts/checkpoints/h1_seed<s>_early_stop.pt` (trọng số MLP + danh sách cha), đổi thành bảng CPD của $Y$, gắn vào bản sao B0.
* **H2**: đọc sơ đồ từ `artifacts/graphs/h2_dag_seed<s>.json`, điền bảng BDeu như B0.

Sau đó chấm lại trên **đúng tập TEST** của seed đó (file split đã lưu) và so với số trong **Table A** (AUROC) và **Table B** (ATE). Khớp nghĩa là các file đã lưu tái tạo đúng kết quả của notebook chính.'''))

C.append(nbf.v4.new_code_cell(r'''import itertools, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn import metrics
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
pd.set_option('display.width', None); pd.set_option('display.max_columns', None)

GOC = Path.cwd()
while not (GOC / 'artifacts' / 'checkpoints').exists() and GOC != GOC.parent:
    GOC = GOC.parent
A = GOC / 'artifacts'
NODES = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'T', 'Y']
STATE_CARD = {'Z1': 3, 'Z2': 2, 'Z3': 2, 'Z4': 2, 'Z5': 2, 'T': 2, 'Y': 2}
SEEDS = [42, 43, 44, 45, 46]

# Mã hóa 7 biến đúng như mục 8.A của notebook chính
raw = pd.read_csv(GOC / 'data' / 'uci_697.csv', sep=None, engine='python')
raw.columns = [c.strip().lstrip('﻿') for c in raw.columns]
d = raw[raw['Target'].isin(['Dropout', 'Graduate'])].reset_index(drop=True)
DF = pd.DataFrame({
    'Z1': pd.cut(pd.to_numeric(d['Admission grade']), bins=[-np.inf, 120.0, 140.0, np.inf], labels=[0, 1, 2]).astype(int),
    'Z2': d['Debtor'].astype(int), 'Z3': d['Tuition fees up to date'].astype(int),
    'Z4': (pd.to_numeric(d["Mother's qualification"]) >= 13).astype(int),
    'Z5': (pd.to_numeric(d['Age at enrollment']) > 20).astype(int),
    'T': d['Scholarship holder'].astype(int), 'Y': (d['Target'] == 'Graduate').astype(int)})
print('Dữ liệu:', DF.shape, '| thư mục dự án:', GOC)'''))

C.append(nbf.v4.new_code_cell(r'''def fit_bn(edges, train):
    """Dựng mạng Bayes từ danh sách cạnh đã lưu và điền CPD BDeu (ESS = 5) trên TRAIN — giống fit_bn_on_dag."""
    bn = DiscreteBayesianNetwork([tuple(e) for e in edges]); bn.add_nodes_from(NODES)
    bn.fit(train[NODES], estimator=DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=5))
    assert bn.check_model()
    return bn

def query(bn, df):
    """q(0, C_i), q(1, C_i) và xác suất dự báo — giống query_model_effects."""
    cpd = bn.get_cpds('Y'); par = cpd.variables[1:]; n = len(df)
    i0, i1, ip = [1], [1], [1]
    for p in par:
        if p == 'T':
            i0.append(np.zeros(n, int)); i1.append(np.ones(n, int)); ip.append(df['T'].values.astype(int))
        else:
            v = df[p].values.astype(int); i0.append(v); i1.append(v); ip.append(v)
    return cpd.values[tuple(i0)], cpd.values[tuple(i1)], cpd.values[tuple(ip)]

class H1Net(torch.nn.Module):
    """Cùng kiến trúc với NeuralCPDModule để nạp state_dict từ checkpoint."""
    def __init__(self, n_in, h=32, p=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(p),
                                       torch.nn.Linear(h, h // 2), torch.nn.ReLU(), torch.nn.Linear(h // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def encode(df, parents):
    khoi = [np.eye(STATE_CARD[p])[df[p].values.astype(int)] if STATE_CARD[p] > 2 else df[[p]].values for p in parents]
    return torch.tensor(np.hstack(khoi).astype(np.float32))

def h1_from_checkpoint(b0, path):
    """Nạp .pt → MLP (eval) → liệt kê mọi cấu hình cha → bảng CPD của Y → gắn vào bản sao B0 (giống build_h1_bayesian_network)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']; net = H1Net(meta['in_features']); net.load_state_dict(ck['state_dict']); net.eval()
    cpd0 = b0.get_cpds('Y'); par = list(cpd0.variables[1:])
    assert par == list(meta['parents']), (par, meta['parents'])
    combos = pd.DataFrame(list(itertools.product(*[cpd0.state_names[p] for p in par])), columns=par)
    with torch.no_grad():
        p1 = np.clip(torch.sigmoid(net(encode(combos, par))).numpy().astype(np.float64), 1e-4, 1 - 1e-4)
    cpd = TabularCPD('Y', 2, np.vstack([1 - p1, p1]), evidence=par, evidence_card=list(cpd0.cardinality[1:]),
                     state_names={v: cpd0.state_names[v] for v in cpd0.variables})
    h1 = DiscreteBayesianNetwork(list(b0.edges())); h1.add_nodes_from(NODES)
    for c in b0.get_cpds():
        if c.variable != 'Y': h1.add_cpds(c.copy())
    h1.add_cpds(cpd); assert h1.check_model()
    return h1, meta'''))

C.append(nbf.v4.new_code_cell(r'''tabA = pd.read_csv(A / 'tables' / 'Table_A_Prediction.csv')
tabB = pd.read_csv(A / 'tables' / 'Table_B_Intervention.csv')
tabA = tabA[tabA['Seed'].astype(str).str.isdigit()].assign(Seed=lambda x: x['Seed'].astype(int))
tabB = tabB[tabB['Seed'].astype(str).str.isdigit()].assign(Seed=lambda x: x['Seed'].astype(int))

dong = []
for s in SEEDS:
    sp = json.load(open(A / 'splits' / f'split_seed_{s}.json', encoding='utf-8'))
    tr, te = DF.iloc[sp['train_indices']].reset_index(drop=True), DF.iloc[sp['test_indices']].reset_index(drop=True)
    b0 = fit_bn(json.load(open(A / 'graphs' / f'b0_dag_seed{s}.json', encoding='utf-8'))['edges'], tr)
    h1, meta = h1_from_checkpoint(b0, A / 'checkpoints' / f'h1_seed{s}_early_stop.pt')
    h2 = fit_bn(json.load(open(A / 'graphs' / f'h2_dag_seed{s}.json', encoding='utf-8'))['edges'], tr)
    for ten, bn in [('B0', b0), ('H1', h1), ('H2', h2)]:
        q0, q1, p = query(bn, te)
        auc, ate = metrics.roc_auc_score(te['Y'], p), float(np.mean(q1) - np.mean(q0))
        auc_cu = float(tabA[(tabA.Model == ten) & (tabA.Seed == s)]['AUROC'].iloc[0])
        ate_cu = float(tabB[(tabB.Model == ten) & (tabB.Seed == s)]['ATE'].iloc[0])
        dong.append({'Seed': s, 'Model': ten, 'Nguồn': {'B0': 'b0_dag JSON', 'H1': 'h1 .pt', 'H2': 'h2_dag JSON'}[ten],
                     'AUROC_nạp_lại': auc, 'AUROC_Table_A': auc_cu, 'Lệch_AUROC': auc - auc_cu,
                     'ATE_nạp_lại': ate, 'ATE_Table_B': ate_cu, 'Lệch_ATE': ate - ate_cu})
kq = pd.DataFrame(dong)
display(kq.round(6))
lech = kq[['Lệch_AUROC', 'Lệch_ATE']].abs().max()
print(f"Lệch lớn nhất — AUROC: {lech['Lệch_AUROC']:.2e} | ATE: {lech['Lệch_ATE']:.2e}")
print('KẾT LUẬN:', 'KHỚP — file đã lưu tái tạo đúng Table A, B' if lech.max() < 1e-6 else 'CÓ CHÊNH LỆCH — xem bảng trên')'''))

C.append(nbf.v4.new_code_cell(r'''# Thông tin trong checkpoint H1 (seed 42) — những gì Slide 60 yêu cầu ghi lại
ck = torch.load(A / 'checkpoints' / 'h1_seed42_early_stop.pt', map_location='cpu', weights_only=False)
display(pd.Series({k: v for k, v in ck['meta'].items() if k not in ('train_loss', 'val_loss')}, name='h1_seed42_early_stop.pt').to_frame())
print('Các khóa trọng số:', list(ck['state_dict'].keys()))'''))

nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
nbf.write(nb, open(sys.argv[1], 'w', encoding='utf-8'))
print('written', sys.argv[1])
