# -*- coding: utf-8 -*-
"""Bản code CÓ CHÚ THÍCH TỪNG DÒNG (# ở cuối dòng) cho các hàm mục 8–9.
Bỏ hết comment đi thì phải giống hệt code trong notebook (gen_groups.py kiểm tra bằng so sánh AST)."""

A = {}

A['temporal_forbidden_edges'] = r'''def temporal_forbidden_edges(cfg: Config) -> List[Tuple[str, str]]:
    """
    Danh sách cạnh cấm theo trật tự thời gian Z (nhập học) -> T (học bổng) -> Y (kết cục), dùng CHUNG cho B0 và H2:
    cấm Y -> (T, Z) và T -> Z.
    """
    cam = [(cfg.outcome, v) for v in cfg.all_nodes if v != cfg.outcome]  # cấm Y -> mọi nút khác (kết cục không gây ra quá khứ)
    cam += [(cfg.treatment, z) for z in cfg.confounders]  # cấm T -> Z1..Z5 (học bổng không gây ra biến nền)
    return cam  # danh sách cặp (nguồn, đích) bị cấm; cạnh Z -> Z KHÔNG bị cấm
'''

A['validation_log_likelihood'] = r'''def validation_log_likelihood(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> float:
    """Log-likelihood của toàn mạng trên một tập dữ liệu: sum_i sum_v log P(x_v | pa(x_v))."""
    tong = 0.0  # tổng log-xác suất, càng gần 0 (ít âm) càng tốt
    for nut in model.nodes():  # duyệt từng nút của mạng (7 nút)
        cpd = model.get_cpds(nut)  # bảng xác suất của nút: P(nút | cha của nút)
        chi_so = tuple(np.searchsorted(np.array(cpd.state_names[v]), df[v].values) for v in cpd.variables)  # đổi giá trị của nút + cha thành chỉ số ô trong bảng, cho mọi dòng
        tong += float(np.log(cpd.values[chi_so]).sum())  # tra xác suất từng dòng, lấy log, cộng dồn
    return tong  # = log P(cả tập dữ liệu | mô hình)
'''

A['learn_b0_dag'] = r'''def learn_b0_dag(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config) -> Tuple[nx.DiGraph, str, List[Dict[str, Any]]]:
    """
    Học DAG B0 trên TRAIN bằng pgmpy HillClimbSearch (Slide 57, 59):
      - ràng buộc: cạnh cấm theo thời gian + cạnh bắt buộc T -> Y (giả thuyết cần ước lượng);
      - lưới hàm điểm cfg.b0_score_grid; chọn hàm điểm có log-likelihood VALIDATION cao nhất, rồi đóng băng.
    Trả ra: DAG được chọn, tên hàm điểm, bảng các ứng viên.
    """
    tri_thuc = ExpertKnowledge(forbidden_edges=temporal_forbidden_edges(cfg), required_edges=list(cfg.required_edges))  # "luật chơi": cạnh cấm + cạnh bắt buộc T -> Y
    du_lieu = train_df[list(cfg.all_nodes)]  # chỉ 7 cột Z1..Z5, T, Y của TRAIN
    ham_diem = {'BIC': BIC(du_lieu), 'BDeu': BDeu(du_lieu, equivalent_sample_size=cfg.b0_ess), 'K2': K2(du_lieu)}  # 3 "giám khảo" chấm sơ đồ
    ung_vien: List[Dict[str, Any]] = []  # bảng 3 ứng viên để báo cáo
    tot_nhat: Optional[Tuple[float, str, nx.DiGraph]] = None  # (điểm VAL, tên giám khảo, sơ đồ) tốt nhất đến hiện tại
    for ten in cfg.b0_score_grid:  # lần lượt 'BIC', 'BDeu', 'K2'
        uoc_luong = HillClimbSearch(du_lieu).estimate(scoring_method=ham_diem[ten], expert_knowledge=tri_thuc, show_progress=False)  # leo đồi: thêm/bớt/đảo cạnh theo giám khảo này
        dag = nx.DiGraph(list(uoc_luong.edges()))  # chuyển sang đồ thị networkx
        dag.add_nodes_from(cfg.all_nodes)  # thêm nút cô lập (nếu có) để đủ 7 nút
        ll_val = validation_log_likelihood(fit_bn_on_dag(train_df, dag, cfg), val_df)  # điền bảng BDeu trên TRAIN rồi "thi lại" trên VAL
        ung_vien.append({'Score': ten, 'Edges': len(dag.edges()), 'Val_LogLik': ll_val,
                         'Pa(Y)': sorted(dag.predecessors(cfg.outcome))})  # ghi lại: giám khảo, số cạnh, điểm VAL, cha của Y
        if tot_nhat is None or ll_val > tot_nhat[0]:  # điểm VAL cao hơn (ít âm hơn) thì thay
            tot_nhat = (ll_val, ten, dag)
    assert tot_nhat is not None and nx.is_directed_acyclic_graph(tot_nhat[2])  # lưới an toàn: phải là DAG
    for (u, v) in temporal_forbidden_edges(cfg):
        assert not tot_nhat[2].has_edge(u, v), f'Vi phạm ràng buộc thời gian: {u}->{v}'  # lưới an toàn: không có cạnh cấm
    return tot_nhat[2], tot_nhat[1], ung_vien  # sơ đồ được chọn, tên giám khảo thắng (K2), bảng ứng viên
'''

A['query_model_effects'] = r'''def query_model_effects(model: DiscreteBayesianNetwork, df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truy vấn vector hóa từ CPD của Y:
    - q0, q1: q_m(t, C_i) = P_m(Y=1 | T=t, Pa(Y)\\{T} = C_i) — bằng P(Y=1 | do(T=t), C_i) vì trong DAG đã học
      Pa(Y)\\{T} gồm các biến tiền can thiệp (mặt nạ cấm T->Z) nên là một tập điều chỉnh cửa sau hợp lệ;
    - pred: P(Y=1 | T_obs, Pa(Y)) cho dự báo ngoài mẫu.
    """
    cpd_y = model.get_cpds(cfg.outcome)  # bảng xác suất của Y (tốt nghiệp)
    parents = cpd_y.variables[1:]  # cha của Y, ví dụ ['T', 'Z2', 'Z3', 'Z5']
    n_rows = len(df)  # số hồ sơ cần tra (545 với TEST)
    if not parents:  # trường hợp hiếm: Y không có cha
        p = float(cpd_y.values[1])  # chỉ có một xác suất chung
        return np.full(n_rows, p), np.full(n_rows, p), np.full(n_rows, p)
    idx_0, idx_1, idx_pred = [1], [1], [1]  # chiều đầu của bảng = trạng thái của Y; chọn 1 = "tốt nghiệp"
    for p in parents:  # dựng chỉ số tra bảng cho từng biến cha
        if p == cfg.treatment:  # cha là học bổng T
            idx_0.append(np.zeros(n_rows, dtype=int))  # q0: ÉP T = 0 cho mọi hồ sơ
            idx_1.append(np.ones(n_rows, dtype=int))  # q1: ÉP T = 1 cho mọi hồ sơ
            idx_pred.append(df[cfg.treatment].values.astype(int))  # dự báo: dùng T thật của hồ sơ
        else:  # cha là biến nền
            vals = df[p].values.astype(int)  # giữ nguyên giá trị thật của hồ sơ
            idx_0.append(vals)
            idx_1.append(vals)
            idx_pred.append(vals)
    q0 = cpd_y.values[tuple(idx_0)].astype(np.float64)  # P(Y=1 | do(T=0), hồ sơ i) cho từng i
    q1 = cpd_y.values[tuple(idx_1)].astype(np.float64)  # P(Y=1 | do(T=1), hồ sơ i)
    pred_probs = cpd_y.values[tuple(idx_pred)].astype(np.float64)  # P(Y=1 | T thật, hồ sơ i) — để chấm AUROC
    return q0, q1, pred_probs  # ATE = mean(q1) - mean(q0); không dùng Y thật ở đây
'''

A['NeuralCPDModule'] = r'''class NeuralCPDModule(nn.Module):
    """MLP ước lượng P(Y=1 | Pa(Y)); số chiều đầu vào phụ thuộc tập cha Pa(Y) của DAG B0 trong từng run."""
    def __init__(self, in_features: int, hidden_dim: int = 32, dropout: float = 0.2):  # in_features = số chiều đầu vào (4 khi Pa(Y) gồm 4 biến nhị phân)
        super().__init__()
        self.net = nn.Sequential(  # "net" = chuỗi các lớp của mạng, chạy lần lượt từ trên xuống
            nn.Linear(in_features, hidden_dim),  # lớp 1: 4 -> 32 nơ-ron (trọng số + độ lệch)
            nn.BatchNorm1d(hidden_dim),  # chuẩn hóa theo lô: giữ giá trị trong khoảng ổn định, học nhanh hơn
            nn.ReLU(),  # hàm kích hoạt: giữ số dương, số âm thành 0 (tạo tính phi tuyến)
            nn.Dropout(dropout),  # khi huấn luyện, tắt ngẫu nhiên 20% nơ-ron để chống học thuộc
            nn.Linear(hidden_dim, hidden_dim // 2),  # lớp 2: 32 -> 16 nơ-ron
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # lớp ra: 16 -> 1 số (logit, chưa phải xác suất)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: ma trận [số dòng, số chiều đầu vào]
        return self.net(x).squeeze(-1)  # trả logit; xác suất = sigmoid(logit)
'''

A['encode_h1_features'] = r'''def encode_h1_features(df: pd.DataFrame, parents: List[str]) -> torch.Tensor:
    """Mã hóa DUY NHẤT các nút cha của Y (Slide 60): biến 3 trạng thái -> one-hot, biến nhị phân giữ nguyên."""
    khoi: List[np.ndarray] = []  # mỗi cha một khối cột
    for p in parents:
        if STATE_CARD[p] > 2:  # biến có 3 trạng thái (Z1 điểm vào: 0/1/2)
            khoi.append(np.eye(STATE_CARD[p])[df[p].values.astype(int)])  # one-hot: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
        else:  # biến nhị phân 0/1: giữ nguyên một cột
            khoi.append(df[[p]].values.astype(np.float32))
    return torch.tensor(np.hstack(khoi).astype(np.float32), dtype=torch.float32)  # ghép các khối thành ma trận đầu vào cho MLP
'''

A['train_h1_mlp'] = r'''def train_h1_mlp(train_df: pd.DataFrame, val_df: pd.DataFrame, parents: List[str], cfg: Config,
                 seed: int, dung_som: bool = True) -> Tuple[NeuralCPDModule, Dict[str, Any]]:
    """Huấn luyện MLP trên TRAIN (BCE), theo dõi loss VALIDATION sau mỗi epoch; trả về mô hình và nhật ký.
    dung_som=True (mặc định, dùng cho Tables A–C): dừng khi VAL không cải thiện cfg.h1_patience epoch và khôi phục
    trọng số tốt nhất. dung_som=False: chạy đủ cfg.h1_epochs và giữ trọng số của epoch cuối (để so sánh, mục 8.E2)."""
    set_seed(seed)  # cố định ngẫu nhiên: cùng seed -> cùng kết quả
    X_train = encode_h1_features(train_df, parents)  # đầu vào TRAIN (chỉ cha của Y)
    y_train = torch.tensor(train_df['Y'].values, dtype=torch.float32)  # nhãn TRAIN 0/1
    X_val = encode_h1_features(val_df, parents)  # đầu vào VAL
    y_val = torch.tensor(val_df['Y'].values, dtype=torch.float32)  # nhãn VAL
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.h1_batch_size, shuffle=True)  # chia TRAIN thành lô 64 dòng, xáo mỗi epoch
    model = NeuralCPDModule(h1_input_dim(parents), cfg.h1_hidden_dim, cfg.h1_dropout)  # mạng 769 tham số
    criterion = nn.BCEWithLogitsLoss()  # hàm mất mát: sigmoid + binary cross-entropy (phạt đoán sai, nhất là sai mà tự tin)
    optimizer = optim.Adam(model.parameters(), lr=cfg.h1_lr, weight_decay=cfg.h1_weight_decay)  # Adam: bước học 0.005; weight_decay phạt trọng số lớn
    best_val, patience, best_w, best_epoch = float('inf'), 0, None, 0  # kỷ lục loss VAL, bộ đếm "không tiến bộ", trọng số tốt nhất, epoch tốt nhất
    train_hist: List[float] = []  # lịch sử loss TRAIN từng epoch
    val_hist: List[float] = []  # lịch sử loss VAL từng epoch
    for epoch in range(cfg.h1_epochs):  # tối đa 150 epoch
        model.train()  # chế độ huấn luyện: bật dropout
        tong, dem = 0.0, 0  # cộng dồn loss và số dòng trong epoch
        for bx, by in loader:  # từng lô 64 dòng
            optimizer.zero_grad()  # xóa gradient của bước trước
            loss = criterion(model(bx), by)  # đoán -> tính loss của lô
            loss.backward()  # lan truyền ngược: tính gradient
            optimizer.step()  # cập nhật trọng số theo gradient
            tong += loss.item() * len(by)
            dem += len(by)
        model.eval()  # chế độ đánh giá: tắt dropout
        with torch.no_grad():  # không cần gradient khi chỉ đo
            v_loss = criterion(model(X_val), y_val).item()  # loss trên VAL sau epoch này
        train_hist.append(tong / dem)  # loss TRAIN trung bình của epoch
        val_hist.append(v_loss)
        if v_loss < best_val:  # VAL tốt hơn kỷ lục
            best_val, best_epoch, patience = v_loss, epoch, 0  # ghi kỷ lục, đặt lại bộ đếm (epoch đếm từ 0)
            best_w = {k: v.clone() for k, v in model.state_dict().items()}  # sao lưu trọng số tốt nhất
        else:
            patience += 1  # thêm một epoch không tiến bộ
            if dung_som and patience >= cfg.h1_patience:  # 15 epoch liên tiếp không tiến bộ
                break  # early stopping
    if dung_som:
        model.load_state_dict(best_w)  # lấy lại trọng số của epoch tốt nhất
    model.eval()
    nhat_ky = {'parents': parents, 'in_features': h1_input_dim(parents), 'early_stopping': dung_som, 'best_epoch': best_epoch,
               'best_val_bce': best_val, 'epochs_run': len(val_hist), 'train_loss': train_hist, 'val_loss': val_hist,
               'n_params': int(sum(p.numel() for p in model.parameters())),
               'architecture': f'Linear({h1_input_dim(parents)},{cfg.h1_hidden_dim})-BN-ReLU-Dropout({cfg.h1_dropout})-'
                               f'Linear({cfg.h1_hidden_dim},{cfg.h1_hidden_dim // 2})-ReLU-Linear(.,1)',
               'optimizer': f'Adam(lr={cfg.h1_lr}, weight_decay={cfg.h1_weight_decay}), batch={cfg.h1_batch_size}'}  # nhật ký lưu vào checkpoint (Slide 60)
    return model, nhat_ky
'''

A['build_h1_bayesian_network'] = r'''def build_h1_bayesian_network(b0_model: DiscreteBayesianNetwork, h1_mlp: NeuralCPDModule,
                              cfg: Config) -> DiscreteBayesianNetwork:
    """Liệt kê mọi cấu hình cha của Y trong B0, dự đoán bằng MLP, tạo TabularCPD chuẩn hóa và thay vào bản sao B0."""
    b0_y_cpd = b0_model.get_cpds(cfg.outcome)  # bảng của Y trong B0 (để lấy tên cha, số trạng thái)
    parent_vars = list(b0_y_cpd.variables[1:])  # cha của Y
    parent_cards = list(b0_y_cpd.cardinality[1:])  # số trạng thái của từng cha (2 hoặc 3)
    combos = list(itertools.product(*[b0_y_cpd.state_names[p] for p in parent_vars]))  # mọi tổ hợp cha (16 với 4 biến nhị phân)
    df_cfg = pd.DataFrame(combos, columns=parent_vars)  # mỗi dòng một tổ hợp
    with torch.no_grad():
        p_y1 = torch.sigmoid(h1_mlp(encode_h1_features(df_cfg, parent_vars))).numpy().astype(np.float64)  # MLP đoán P(Y=1) cho từng tổ hợp
    p_y1 = np.clip(p_y1, 1e-4, 1.0 - 1e-4)  # chặn số học (khác với làm mịn CPD) để tránh log-loss vô cực
    h1_cpd = TabularCPD(variable=cfg.outcome, variable_card=2, values=np.vstack([1.0 - p_y1, p_y1]),
                        evidence=parent_vars, evidence_card=parent_cards,
                        state_names={v: b0_y_cpd.state_names[v] for v in b0_y_cpd.variables})  # đổi MLP thành bảng CPD: hàng 0 = P(Y=0), hàng 1 = P(Y=1)
    h1_bn = DiscreteBayesianNetwork(list(b0_model.edges()))  # mạng mới, CÙNG sơ đồ với B0
    h1_bn.add_nodes_from(cfg.all_nodes)
    for cpd in b0_model.get_cpds():
        if cpd.variable != cfg.outcome:
            h1_bn.add_cpds(cpd.copy())  # giữ nguyên bảng của mọi nút khác Y
    h1_bn.add_cpds(h1_cpd)  # chỉ bảng của Y là từ mạng nơ-ron
    assert h1_bn.check_model(), 'Lỗi: check_model() thất bại cho H1!'  # kiểm tra mỗi bảng hợp lệ (tổng = 1, khớp cha)
    return h1_bn  # H1 = B0 với CPD(Y) nơ-ron -> truy vấn bằng đúng query_model_effects
'''

A['build_domain_mask'] = r'''def build_domain_mask(cfg: Config) -> np.ndarray:
    """
    Mặt nạ M (Slide 63): M[i, j] = 0 nếu cạnh i -> j bị cấm. Dùng ĐÚNG danh sách cạnh cấm của B0
    (temporal_forbidden_edges) + cấm tự nối. Quy ước W[i, j] nghĩa là cạnh i -> j.
    """
    nodes = list(cfg.all_nodes)  # thứ tự nút cố định: Z1..Z5, T, Y
    M = np.ones((len(nodes), len(nodes)), dtype=np.float32)  # bắt đầu: cho phép mọi cạnh (1)
    np.fill_diagonal(M, 0.0)  # cấm tự nối i -> i
    for (u, v) in temporal_forbidden_edges(cfg):  # cùng danh sách cấm với B0
        M[nodes.index(u), nodes.index(v)] = 0.0  # 0 = cạnh u -> v bị cấm
    return M  # ma trận 7 x 7 gồm 0/1
'''

A['NotearsMLP'] = r'''class NotearsMLP(nn.Module):
    """
    NOTEARS phi tuyến: mỗi nút j có một MLP riêng f_j(X) với tầng ẩn sigmoid m nơ-ron.
    Trọng số cạnh W[i, j] = || trọng số tầng 1 từ đầu vào i vào MLP của nút j ||_2.
    Mặt nạ M nhân trực tiếp vào trọng số tầng 1 nên cạnh bị cấm có gradient bằng 0 và luôn bằng 0.
    """
    def __init__(self, d: int, hidden: int, mask: np.ndarray):  # d = số nút (7), hidden = nơ-ron ẩn mỗi mạng con (10)
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))  # lưu mặt nạ M (không phải tham số học)
        self.fc1 = nn.Parameter(torch.randn(d, hidden, d) * 0.1)   # [nút j, nơ-ron ẩn, đầu vào i]
        self.b1 = nn.Parameter(torch.zeros(d, hidden))  # độ lệch lớp 1 của từng mạng con
        self.fc2 = nn.Parameter(torch.randn(d, hidden) * 0.1)  # trọng số lớp 2: 10 nơ-ron ẩn -> 1 đầu ra, cho từng nút
        self.b2 = nn.Parameter(torch.zeros(d))  # độ lệch đầu ra của từng nút

    def masked_fc1(self) -> torch.Tensor:
        return self.fc1 * self.mask.T.unsqueeze(1)  # nhân mặt nạ: trọng số từ đầu vào bị cấm luôn = 0 ("bịt tai")

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X: [số dòng n, d biến]
        h = torch.sigmoid(torch.einsum('jmi,ni->njm', self.masked_fc1(), X) + self.b1)  # lớp ẩn của mọi mạng con cùng lúc: [n, nút j, nơ-ron m]
        return torch.einsum('njm,jm->nj', h, self.fc2) + self.b2  # dự đoán lại từng biến: [n, d]

    def adjacency_sq(self) -> torch.Tensor:
        """(W[i, j])^2 — khả vi tại 0, dùng trong ràng buộc h(W) = tr(exp(W∘W)) - d."""
        return (self.masked_fc1() ** 2).sum(dim=1).T  # cộng bình phương qua nơ-ron ẩn -> độ mạnh^2 của cạnh i -> j
'''

A['train_notears_mlp'] = r'''def train_notears_mlp(train_df: pd.DataFrame, cfg: Config, seed: int, lambda_l1: float) -> Dict[str, Any]:
    """
    Augmented Lagrangian: vòng trong tối ưu Adam (cfg.h2_inner_iters bước), vòng ngoài cập nhật alpha và tăng rho x10
    khi h không giảm đủ 4 lần; dừng khi h < cfg.h2_h_tol (hội tụ) hoặc hết cfg.h2_max_outer vòng (báo KHÔNG hội tụ).
    """
    set_seed(seed)  # cố định ngẫu nhiên
    nodes = list(cfg.all_nodes)  # thứ tự nút Z1..Z5, T, Y
    d = len(nodes)  # số biến = 7
    mask = build_domain_mask(cfg)  # mặt nạ cạnh cấm (cùng luật với B0)
    X = torch.tensor(train_df[nodes].values, dtype=torch.float32)  # dữ liệu TRAIN dạng tensor [2541, 7]
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)  # chuẩn hóa từng cột: trung bình 0, độ lệch chuẩn 1 (+1e-6 tránh chia 0)
    net = NotearsMLP(d, cfg.h2_hidden, mask)  # "net" = mạng NOTEARS: 7 mạng con, mỗi mạng đoán một biến từ các biến khác
    rho, alpha, h_prev = 1.0, 0.0, float('inf')  # rho: hệ số phạt BẬC HAI cho vòng tròn (tăng x10 khi cần); alpha: hệ số phạt TUYẾN TÍNH (nhân tử Lagrange); h_prev: h của vòng trước
    h_hist: List[float] = []  # log h(W) sau mỗi vòng ngoài
    loss_hist: List[float] = []  # log sai số tái tạo sau mỗi vòng ngoài
    for _ in range(cfg.h2_max_outer):  # VÒNG NGOÀI: tối đa 12 lần siết ràng buộc
        while rho < cfg.h2_rho_max:  # lặp đến khi h giảm đủ (hoặc rho quá lớn)
            opt = optim.Adam(net.parameters(), lr=cfg.h2_lr)  # bộ tối ưu Adam, bước học 0.01
            for _ in range(cfg.h2_inner_iters):  # VÒNG TRONG: 300 bước gradient
                opt.zero_grad()  # xóa gradient cũ
                h = torch.trace(torch.matrix_exp(net.adjacency_sq())) - d  # "máy dò vòng tròn": h = 0 khi và chỉ khi không có chu trình
                recon = 0.5 * ((X - net(X)) ** 2).mean()  # sai số tái tạo: mạng đoán lại dữ liệu kém cỡ nào
                loss = recon + lambda_l1 * net.masked_fc1().abs().sum() + alpha * h + 0.5 * rho * h * h  # tổng phạt = khớp dữ liệu + phạt nhiều cạnh (L1) + phạt vòng tròn (alpha*h + rho/2*h^2)
                loss.backward()  # tính gradient của tổng phạt
                opt.step()  # cập nhật trọng số
            with torch.no_grad():
                h_new = float((torch.trace(torch.matrix_exp(net.adjacency_sq().double())) - d).item())  # đo lại h sau 300 bước (độ chính xác double)
            if h_new > 0.25 * h_prev:  # h chưa giảm xuống dưới 1/4 lần trước
                rho *= 10.0  # phạt vòng tròn nặng gấp 10 rồi chạy lại vòng trong
            else:
                break  # h giảm đủ -> sang bước cập nhật vòng ngoài
        h_prev = h_new  # nhớ h để so ở vòng sau
        alpha += rho * h_new  # cập nhật nhân tử Lagrange: còn vòng tròn thì phạt tuyến tính tăng dần
        h_hist.append(h_new)  # ghi log
        loss_hist.append(float(recon.item()))
        if h_new < cfg.h2_h_tol:  # h < 1e-6: coi như không còn vòng -> hội tụ
            break
    with torch.no_grad():
        W = torch.sqrt(net.adjacency_sq()).numpy().astype(np.float64)  # ma trận độ mạnh cạnh W[i, j] (đã qua mặt nạ)
    forbidden_max = float(np.abs(W[mask == 0]).max())  # kiểm tra tuân thủ: trọng số lớn nhất ở ô bị cấm (phải = 0)
    return {'W': W, 'mask': mask, 'h_final': h_hist[-1], 'converged': bool(h_hist[-1] < cfg.h2_h_tol),
            'outer_iters': len(h_hist), 'h_history': h_hist, 'recon_history': loss_hist,
            'forbidden_weight_max': forbidden_max, 'lambda_l1': lambda_l1}  # kết quả + log để lưu JSON
'''

A['graph_from_weights'] = r'''def graph_from_weights(W: np.ndarray, threshold: float, cfg: Config) -> Tuple[nx.DiGraph, List[Tuple[str, str]], bool]:
    """
    Lọc ngưỡng |W| >= threshold; nếu còn chu trình thì cắt cạnh có trọng số nhỏ nhất trong chu trình (luật sửa khai báo
    trước, Slide 79); áp cùng cạnh bắt buộc T -> Y như B0. Trả ra DAG, các cạnh bị cắt, và cờ 'T->Y bị ép thêm'.
    """
    nodes = list(cfg.all_nodes)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from([(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(len(nodes)) if W[i, j] >= threshold])  # giữ cạnh có độ mạnh >= tau ("chấm đậu")
    cut: List[Tuple[str, str]] = []  # cạnh bị cắt để phá vòng (thực tế không có)
    while not nx.is_directed_acyclic_graph(g):  # nếu còn chu trình
        cyc = nx.find_cycle(g)  # tìm một vòng
        u, v = min(cyc, key=lambda e: W[nodes.index(e[0]), nodes.index(e[1])])[:2]  # cạnh yếu nhất trong vòng
        g.remove_edge(u, v)  # cắt nó
        cut.append((u, v))
    forced = False  # có phải ép thêm T -> Y không
    for (u, v) in cfg.required_edges:
        if not g.has_edge(u, v):
            g.add_edge(u, v)  # thêm T -> Y như B0 nếu thiếu
            forced = True
    return g, cut, forced
'''

A['select_h2_graph'] = r'''def select_h2_graph(res: Dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame,
                    cfg: Config) -> Tuple[nx.DiGraph, float, List[Dict[str, Any]]]:
    """Chọn ngưỡng cạnh trên VALIDATION (log-likelihood của mạng BDeu khớp trên TRAIN), rồi đóng băng (Slide 62)."""
    bang: List[Dict[str, Any]] = []  # bảng các mức tau đã thử
    tot: Optional[Tuple[float, float, nx.DiGraph]] = None  # (điểm VAL, tau, sơ đồ) tốt nhất
    for tau in cfg.h2_threshold_grid:  # thử tau = 0.1, 0.2, 0.3, 0.5, 0.8
        g, cut, forced = graph_from_weights(res['W'], tau, cfg)  # lọc W thành sơ đồ
        ll = validation_log_likelihood(fit_bn_on_dag(train_df, g, cfg), val_df)  # điền bảng BDeu trên TRAIN, chấm trên VAL
        bang.append({'tau': tau, 'Edges': len(g.edges()), 'Cut_for_cycle': cut, 'T->Y_forced': forced,
                     'Val_LogLik': ll, 'Pa(Y)': sorted(g.predecessors(cfg.outcome))})
        if tot is None or ll > tot[0]:
            tot = (ll, tau, g)
    return tot[2], tot[1], bang  # sơ đồ tốt nhất, tau được chọn, bảng để báo cáo
'''

A['evaluate_prediction_metrics'] = r'''def evaluate_prediction_metrics(y_true: np.ndarray, y_prob: np.ndarray, val_y_true: np.ndarray,
                                val_y_prob: np.ndarray) -> Dict[str, float]:
    """AUROC, Brier, log-loss và Macro-F1 với ngưỡng F1 chọn trên VALIDATION (Slide 74)."""
    best_thresh, best_f1 = 0.5, -1.0  # ngưỡng cắt "đoán tốt nghiệp" và F1 tốt nhất
    for th in np.linspace(0.2, 0.8, 31):  # thử 31 ngưỡng từ 0.2 đến 0.8
        f1 = metrics.f1_score(val_y_true, (val_y_prob >= th).astype(int), average='macro', zero_division=0)  # Macro-F1 trên VAL
        if f1 > best_f1:
            best_f1, best_thresh = f1, th
    y_pred = (y_prob >= best_thresh).astype(int)  # áp ngưỡng chọn trên VAL cho TEST
    return {
        'AUROC': float(metrics.roc_auc_score(y_true, y_prob)),  # khả năng xếp hạng đúng (0.5 = đoán bừa)
        'Macro_F1': float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)),  # trung bình F1 của hai lớp
        'Brier_Score': float(metrics.brier_score_loss(y_true, y_prob)),  # sai số bình phương của xác suất
        'Log_Loss': float(metrics.log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),  # BCE trên TEST
        'Precision': float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        'F1_Threshold_from_Val': float(best_thresh),
    }
'''

A['compute_bootstrap_ate_ci'] = r'''def compute_bootstrap_ate_ci(q0: np.ndarray, q1: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """
    95% CI percentile bootstrap: lấy mẫu lại các hồ sơ tham chiếu C_test (có hoàn lại), GIỮ CỐ ĐỊNH mô hình đã khớp.
    Đây là bất định do lấy mẫu quần thể tham chiếu, chưa gồm bất định tham số/đồ thị (Slide 68).
    """
    diffs = q1 - q0  # hiệu "có - không học bổng" của từng hồ sơ TEST
    boot = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]  # 1000 lần: bốc lại 545 hồ sơ có hoàn lại, tính ATE
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))  # phân vị 2.5% và 97.5% (alpha = 0.05)
'''

A['overlap_report'] = r'''def overlap_report(train_df: pd.DataFrame, test_df: pd.DataFrame, adj_vars: List[str], cfg: Config) -> Dict[str, Any]:
    """
    Kiểm tra hỗ trợ (Slide 69, 79): một hồ sơ test C_i là 'không được hỗ trợ' nếu trong TRAIN không có ít nhất một
    sinh viên mang cùng cấu hình adj_vars ở CẢ T=0 và T=1.
    """
    if not adj_vars:
        return {'unsupported_test_profiles': 0, 'unsupported_pct': 0.0, 'status': 'no adjustment variables'}
    dem = train_df.groupby(adj_vars + [cfg.treatment]).size().unstack(fill_value=0)  # đếm số người TRAIN theo (hồ sơ, T)
    for t in (0, 1):
        if t not in dem.columns:
            dem[t] = 0
    ho_tro = set(dem[(dem[0] > 0) & (dem[1] > 0)].index.to_list())  # hồ sơ có CẢ người có và không có học bổng
    khoa = [tuple(r) if len(adj_vars) > 1 else r[0] for r in test_df[adj_vars].values.tolist()]  # hồ sơ của từng người TEST
    khong = int(sum(k not in ho_tro for k in khoa))  # số người TEST mà hồ sơ không có đủ người so sánh
    return {'unsupported_test_profiles': khong, 'unsupported_pct': 100.0 * khong / len(test_df),
            'status': 'full overlap' if khong == 0 else f'{khong} test profiles lack overlap'}
'''

A['record_model_run'] = r'''def record_model_run(model_name: str, seed: int, graph_id: str, dag: nx.DiGraph, bn: DiscreteBayesianNetwork,
                     train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                     fit_time: float, cfg: Config) -> Dict[str, Any]:
    """Đánh giá dự báo + can thiệp chuẩn hóa trên C_test cho một mô hình và ghi vào bảng kết quả."""
    t0 = time.time()  # bấm giờ thời gian truy vấn
    _, _, prob_val = query_model_effects(bn, val_df, cfg)  # xác suất dự báo trên VAL (để chọn ngưỡng F1)
    q0, q1, prob_test = query_model_effects(bn, test_df, cfg)  # q0, q1 cho ATE + xác suất dự báo trên TEST
    query_time = time.time() - t0
    ate = float(np.mean(q1) - np.mean(q0))  # ATE = risk(1) - risk(0), KHÔNG dùng Y_test
    lo, hi = compute_bootstrap_ate_ci(q0, q1, n_boot=cfg.n_bootstrap)  # CI 95% bootstrap
    met = evaluate_prediction_metrics(test_df['Y'].values, prob_test, val_df['Y'].values, prob_val)  # chỉ số dự báo, DÙNG Y_test
    adj = [p for p in bn.get_cpds(cfg.outcome).variables[1:] if p != cfg.treatment]  # tập điều chỉnh = cha của Y trừ T
    ident = identify_with_dowhy(dag, train_df, cfg)  # DoWhy kiểm tra định danh trên đúng sơ đồ này
    ov = overlap_report(train_df, test_df, adj, cfg)  # kiểm tra hồ sơ TEST thiếu người so sánh
    prediction_results.append({'Model': model_name, 'Seed': seed, 'Graph': graph_id, 'Test_Size': len(test_df),
                               'Test_Positive_Rate': float(test_df['Y'].mean()), **met,
                               'Fit_Time_s': fit_time, 'Query_Time_s': query_time})  # một dòng của Table A
    intervention_results.append({'Model': model_name, 'Seed': seed, 'Reference_Cohort': f'C_test(seed {seed})',
                                 'Cohort_Size': len(test_df), 'Contrast': 'do(T=1) vs do(T=0)',
                                 'Risk_do_T0': float(np.mean(q0)), 'Risk_do_T1': float(np.mean(q1)),
                                 'ATE': ate, 'CI_95_Low': lo, 'CI_95_High': hi, 'CI_Method': CI_METHOD,
                                 'Formula': 'mean_i P(Y=1 | T=t, Pa(Y)\\T = C_i)',
                                 'Adjustment_Set': '{' + ', '.join(adj) + '}',
                                 'DoWhy_Identification': ident['status'],
                                 'DoWhy_Backdoor_Set': '{' + ', '.join(ident['backdoor_set']) + '}',
                                 'Overlap_Status': ov['status'], 'Unsupported_Test_Profiles_pct': ov['unsupported_pct']})  # một dòng của Table B
    return {'q0': q0, 'q1': q1, 'identification': ident['text']}
'''

A['dowhy_refute_on_graph'] = r'''def dowhy_refute_on_graph(dag: nx.DiGraph, df: pd.DataFrame, cfg: Config, seed: int) -> Dict[str, Any]:
    """
    Refuter DoWhy (Slide 69-70) trên ĐÚNG DAG của mô hình, ước lượng hỗ trợ: backdoor.linear_regression.
    Kết quả này kiểm tra (đồ thị + ước lượng tuyến tính), KHÔNG kiểm tra trực tiếp CPD BDeu hay mạng nơ-ron.
    """
    cm = CausalModel(data=df[list(cfg.all_nodes)], treatment=cfg.treatment, outcome=cfg.outcome, graph=dag)  # mô hình DoWhy với sơ đồ đã học
    estimand = cm.identify_effect(proceed_when_unidentifiable=False)  # tìm tập cửa sau
    est = cm.estimate_effect(estimand, method_name='backdoor.linear_regression', test_significance=False)  # ATE bằng hồi quy tuyến tính (ước lượng hỗ trợ)
    np.random.seed(seed)  # DoWhy dùng bộ sinh số toàn cục -> cố định để tái lập
    plc = cm.refute_estimate(estimand, est, method_name='placebo_treatment_refuter', placebo_type='permute',
                             num_simulations=cfg.placebo_simulations, random_state=seed)  # học bổng giả (xáo T) 30 lần -> kỳ vọng ~ 0
    np.random.seed(seed)
    sub = cm.refute_estimate(estimand, est, method_name='data_subset_refuter', subset_fraction=cfg.subset_fraction,
                             num_simulations=cfg.placebo_simulations, random_state=seed)  # 80% dữ liệu, 30 lần -> kỳ vọng gần gốc
    return {'original': float(est.value), 'placebo': float(plc.new_effect), 'subset': float(sub.new_effect),
            'backdoor': sorted(estimand.get_backdoor_variables())}
'''

A['h1_retraining_refuter'] = r'''def h1_retraining_refuter(b0_model: DiscreteBayesianNetwork, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: pd.DataFrame, cfg: Config, seed: int) -> Dict[str, float]:
    """
    Kiểm định tương tự cho H1 (Slide 70-71: refuter tuyến tính KHÔNG tự động kiểm tra H1 nơ-ron):
    - placebo: hoán vị T trong TRAIN và VALIDATION, huấn luyện lại MLP, tính ATE trên C_test (kỳ vọng ~ 0);
    - subset: huấn luyện lại MLP trên 80% TRAIN, tính ATE trên C_test (kỳ vọng gần ATE gốc).
    """
    parents = list(b0_model.get_cpds('Y').variables[1:])  # cha của Y trong sơ đồ B0
    rng = np.random.default_rng(seed)  # bộ sinh số riêng, cố định seed
    plc, sub = [], []  # ATE của các lần placebo / subset
    for k in range(cfg.h1_refute_sims):  # 3 lần (huấn luyện lại tốn thời gian)
        tr_p, va_p = train_df.copy(), val_df.copy()
        tr_p['T'] = rng.permutation(tr_p['T'].values)  # xáo trộn học bổng trong TRAIN -> học bổng "giả"
        va_p['T'] = rng.permutation(va_p['T'].values)  # và trong VAL
        m_p, _ = train_h1_mlp(tr_p, va_p, parents, cfg, seed=seed + 100 + k)  # huấn luyện lại MLP trên dữ liệu đã xáo
        q0, q1, _ = query_model_effects(build_h1_bayesian_network(b0_model, m_p, cfg), test_df, cfg)
        plc.append(float(np.mean(q1) - np.mean(q0)))  # ATE placebo (kỳ vọng ~ 0)
        idx = rng.choice(len(train_df), size=int(cfg.subset_fraction * len(train_df)), replace=False)  # chọn ngẫu nhiên 80% TRAIN
        m_s, _ = train_h1_mlp(train_df.iloc[idx].reset_index(drop=True), val_df, parents, cfg, seed=seed + 200 + k)  # huấn luyện lại trên 80%
        q0, q1, _ = query_model_effects(build_h1_bayesian_network(b0_model, m_s, cfg), test_df, cfg)
        sub.append(float(np.mean(q1) - np.mean(q0)))  # ATE subset (kỳ vọng gần gốc)
    return {'placebo': float(np.mean(plc)), 'subset': float(np.mean(sub))}  # trung bình 3 lần
'''

A['tinh_rv_cinelli_hazlett'] = r'''def tinh_rv_cinelli_hazlett(du_lieu: pd.DataFrame) -> KetQuaDoNhayCinelli:
    """
    Hàm này để làm gì: Tính Robustness Value (RV) theo Cinelli & Hazlett (2020, Eq. 18) từ hồi quy OLS
    Y ~ T + Z1..Z5 (cùng mã hóa Z như tools/run_advanced_factorial_and_sensitivity.py):
        f = |t| / sqrt(df),  RV_q=1 = 0.5 * (sqrt(f^4 + 4 f^2) - f^2)
        RV_{q=1, alpha=0.05}: thay f bằng f - t*_{0.975, df-1} / sqrt(df - 1).

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        KetQuaDoNhayCinelli: Hệ số, t-stat, bậc tự do, partial R^2 của T với Y, RV và RV_alpha.
    """
    Z = pd.DataFrame({
        'Z1': pd.qcut(du_lieu['Age_at_enrollment'], q=3, labels=[0, 1, 2]).astype(int),  # tuổi chia 3 nhóm đều (thứ tự Z riêng của mục 9.H)
        'Z2': (du_lieu['Admission_grade'] >= du_lieu['Admission_grade'].median()).astype(int),  # điểm vào >= trung vị
        'Z3': (du_lieu['Mothers_qualification'] > 1).astype(int),  # học vấn mẹ > 1
        'Z4': du_lieu['Debtor'].astype(int),  # Debtor
        'Z5': du_lieu['Tuition_fees_up_to_date'].astype(int),  # học phí đúng hạn
    })
    X = np.column_stack([np.ones(len(du_lieu)), du_lieu['Scholarship_holder'].values, Z.values]).astype(float)  # ma trận [hằng số, T, Z1..Z5]
    y = du_lieu['Target_Graduate'].values.astype(float)  # tốt nghiệp 0/1
    beta = np.linalg.lstsq(X, y, rcond=None)[0]  # hồi quy bình phương tối thiểu; beta[1] = hệ số của học bổng
    phan_du = y - X @ beta  # phần dư
    bac_tu_do = int(len(y) - X.shape[1])  # df = số mẫu - số hệ số = 3623
    se = float(np.sqrt(phan_du @ phan_du / bac_tu_do * np.linalg.inv(X.T @ X)[1, 1]))  # sai số chuẩn của hệ số học bổng
    t_stat = float(beta[1] / se)  # thống kê t (14.20)
    f = abs(t_stat) / np.sqrt(bac_tu_do)  # "cỡ hiệu ứng" theo Cinelli–Hazlett
    f_alpha = max(f - sp_stats.t.ppf(0.975, bac_tu_do - 1) / np.sqrt(bac_tu_do - 1), 0.0)  # bản dùng cho mức ý nghĩa 5%
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)  # RV: nhiễu ẩn mạnh cỡ này mới xóa được hiệu ứng (20.97%)
    rv_alpha = 0.5 * (np.sqrt(f_alpha ** 4 + 4 * f_alpha ** 2) - f_alpha ** 2)  # RV để hiệu ứng mất ý nghĩa thống kê (18.37%)
    return KetQuaDoNhayCinelli(float(beta[1]), t_stat, bac_tu_do,
                               float(t_stat ** 2 / (t_stat ** 2 + bac_tu_do)), float(rv), float(rv_alpha))  # partial R^2 = t^2 / (t^2 + df)
'''

# Đoạn mã tuần tự (không phải hàm) — chú thích phần thân, gen_groups tự bọc vào hàm doan_ma_...
A['__code_75'] = r'''ps_model = LogisticRegression(max_iter=500, random_state=ADV_CONFIG.seed)  # mô hình xác suất được học bổng e(x)
ps_model.fit(df_full[COV_COLS], df_full['T'])  # học T ~ 7 hiệp biến trên 4,424 SV
ps = np.clip(ps_model.predict_proba(df_full[COV_COLS])[:, 1], 0.05, 0.95)  # e(x) từng SV, chặn [0.05, 0.95] để trọng số 1/e không quá lớn

m0 = LogisticRegression(max_iter=500, random_state=ADV_CONFIG.seed)  # mô hình kết cục cho nhóm KHÔNG học bổng (mu0)
m1 = LogisticRegression(max_iter=500, random_state=ADV_CONFIG.seed)  # mô hình kết cục cho nhóm CÓ học bổng (mu1)
m0.fit(df_full[df_full['T'] == 0][COV_COLS], df_full[df_full['T'] == 0]['Y_retain'])  # học "ở lại" ~ hiệp biến, nhóm T=0
m1.fit(df_full[df_full['T'] == 1][COV_COLS], df_full[df_full['T'] == 1]['Y_retain'])  # nhóm T=1

mu0 = m0.predict_proba(df_full[COV_COLS])[:, 1]  # dự đoán ở lại nếu KHÔNG có học bổng, cho MỌI SV
mu1 = m1.predict_proba(df_full[COV_COLS])[:, 1]  # dự đoán ở lại nếu CÓ học bổng, cho MỌI SV

t_full = df_full['T'].values  # học bổng thật
y_full = df_full['Y_retain'].values  # ở lại thật

dr_1 = mu1 + (t_full * (y_full - mu1)) / ps  # điểm AIPW nhánh có học bổng: dự đoán + sửa sai (chỉ người có học bổng), trọng số 1/e
dr_0 = mu0 + ((1.0 - t_full) * (y_full - mu0)) / (1.0 - ps)  # nhánh không học bổng: sửa sai bằng người không học bổng, trọng số 1/(1-e)

risk_t0_ret = float(np.mean(dr_0))  # tỷ lệ ở lại nếu không ai có học bổng (64.28%)
risk_t1_ret = float(np.mean(dr_1))  # nếu ai cũng có học bổng (81.15%)
ate_ret = float(risk_t1_ret - risk_t0_ret)  # ATE giữ chân (+16.87%)

np.random.seed(ADV_CONFIG.seed)  # cố định bootstrap
boot_diffs = []
for _ in range(ADV_CONFIG.n_bootstrap):  # 1000 lần
    idx = np.random.choice(len(dr_1), size=len(dr_1), replace=True)  # bốc lại SV có hoàn lại
    boot_diffs.append(np.mean(dr_1[idx] - dr_0[idx]))  # ATE trên mẫu bốc lại
ci_ret_low = float(np.percentile(boot_diffs, 2.5))  # cận dưới CI 95%
ci_ret_high = float(np.percentile(boot_diffs, 97.5))  # cận trên

print('=== KẾT QUẢ ĐO LƯỜNG TỶ LỆ GIỮ CHÂN TOÀN TRƯỜNG (N = 4,424) ===')
print(f'Risk(do(T=0)) [Tỷ lệ duy trì không học bổng]: {risk_t0_ret*100:.2f}%')
print(f'Risk(do(T=1)) [Tỷ lệ duy trì khi có học bổng]: {risk_t1_ret*100:.2f}%')
print(f'ATE (Mức tăng tỷ lệ duy trì toàn trường): +{ate_ret*100:.2f}%')
print(f'Khoảng tin cậy 95% Bootstrap: [+{ci_ret_low*100:.2f}%, +{ci_ret_high*100:.2f}%]')'''

A['__code_77'] = r'''df_med = df_completed.copy()  # 3,630 SV có kết cục
df_med['M'] = df_med['Curricular units 1st sem (approved)'].astype(float)  # M = số tín chỉ đạt HK1 (biến trung gian)
df_med['Y'] = df_med['Y_grad'].astype(float)  # Y = tốt nghiệp 0/1

X_m = sm.add_constant(df_med[['T'] + COV_COLS])  # [hằng số, học bổng, 7 hiệp biến]
model_m = sm.OLS(df_med['M'], X_m).fit()  # hồi quy M ~ T + hiệp biến
beta_t_m = model_m.params['T']  # Tác động của học bổng lên số tín chỉ đỗ HK1

X_y = sm.add_constant(df_med[['T', 'M'] + COV_COLS])  # [hằng số, học bổng, tín chỉ HK1, 7 hiệp biến]
model_y_lpm = sm.OLS(df_med['Y'], X_y).fit()  # hồi quy Y ~ T + M + hiệp biến (mô hình xác suất tuyến tính)

nde = model_y_lpm.params['T']  # tác dụng TRỰC TIẾP: hệ số của học bổng khi đã giữ M cố định
nie = beta_t_m * model_y_lpm.params['M']  # tác dụng GIÁN TIẾP = (T -> M) x (M -> Y)
te_med = nde + nie  # tổng tác dụng
prop_mediated = (nie / te_med) * 100.0  # % đi qua HK1 (21.5%)

print('=== KẾT QUẢ PHÂN RÃ TRUNG GIAN NHÂN QUẢ (CAUSAL MEDIATION) ===')
print(f'Hệ số T -> M (Tín chỉ HK1 tăng thêm): +{beta_t_m:.3f} tín chỉ (p = {model_m.pvalues["T"]:.2e})')
print(f'Tổng hiệu ứng nhân quả (Total Effect): +{te_med*100:.2f}%')
print(f'Hiệu ứng Trực tiếp Tự nhiên (Natural Direct Effect - NDE): +{nde*100:.2f}% ({100 - prop_mediated:.1f}% tổng hiệu ứng)')
print(f'Hiệu ứng Gián tiếp Tự nhiên (Natural Indirect Effect - NIE): +{nie*100:.2f}% ({prop_mediated:.1f}% tổng hiệu ứng)')
print(f'==> TỶ LỆ TRUNG GIAN QUA ĐÀ HỌC TẬP HK1: {prop_mediated:.1f}%')'''

A['__code_79'] = r'''subgroups_specs = [  # 6 nhóm ĐỊNH TRƯỚC: (tên, điều kiện lọc)
    ('Độ tuổi: Lớn tuổi (> 20 tuổi)', lambda df: df['Z1_age'] > 20),
    ('Độ tuổi: Đúng tuổi (<= 20 tuổi)', lambda df: df['Z1_age'] <= 20),
    ('Tài chính: Đang có Nợ học phí (Debtor = 1)', lambda df: df['Z4_debt'] == 1),
    ('Tài chính: Không nợ học phí (Debtor = 0)', lambda df: df['Z4_debt'] == 0),
    ('Nhà ở: Ở cùng gia đình (Local)', lambda df: df['Z6_disp'] == 0),
    ('Nhà ở: Thuê trọ xa nhà (Displaced)', lambda df: df['Z6_disp'] == 1),
]

cate_rows = []
for label, cond_fn in subgroups_specs:  # từng nhóm
    sub = df_completed[cond_fn(df_completed)].copy()  # SV thuộc nhóm
    active_covs = [c for c in COV_COLS if sub[c].nunique() > 1]  # bỏ hiệp biến không đổi trong nhóm (vd. tuổi khi lọc theo tuổi)
    lr = LogisticRegression(max_iter=500, random_state=ADV_CONFIG.seed)  # mô hình kết cục trong nhóm
    lr.fit(sub[['T'] + active_covs], sub['Y_grad'])  # học tốt nghiệp ~ học bổng + hiệp biến

    sub_0 = sub.copy()
    sub_0['T'] = 0  # ÉP mọi người trong nhóm KHÔNG có học bổng
    sub_1 = sub.copy()
    sub_1['T'] = 1  # ÉP mọi người CÓ học bổng

    p0 = float(np.mean(lr.predict_proba(sub_0[['T'] + active_covs])[:, 1]))  # tỷ lệ tốt nghiệp dự đoán nếu không học bổng
    p1 = float(np.mean(lr.predict_proba(sub_1[['T'] + active_covs])[:, 1]))  # nếu có học bổng (G-computation)
    cate_val = p1 - p0  # CATE của nhóm (hiệu tuyệt đối)

    cate_rows.append({
        'Phân tầng Đối tượng': label,
        'Cỡ mẫu (N)': len(sub),
        'Risk(do(T=0))': f'{p0*100:.2f}%',
        'Risk(do(T=1))': f'{p1*100:.2f}%',
        'Hiệu ứng CATE': f'{cate_val*100:+.2f}%',
        'Tỷ lệ Thay đổi (Relative Risk)': f'{p1 / max(p0, 1e-4):.2f}x'  # tỷ số rủi ro p1/p0 (KHÔNG phải "lợi gấp n lần")
    })

df_cate_display = pd.DataFrame(cate_rows)
print('=== BẢNG HIỆU ỨNG CAN THIỆP KHÔNG ĐỒNG NHẤT (CATE) ===')
display(df_cate_display)'''

A['__code_57'] = r'''ref_seed = 42  # seed tham chiếu cho Table C (seed đầu tiên của 5 run)
split = CACHED_SPLITS[ref_seed]  # chỉ số TRAIN/VAL/TEST đã đóng băng của seed 42
tr42 = PRIMARY_DF.iloc[split['train_indices']].reset_index(drop=True)  # 2,541 SV để học
va42 = PRIMARY_DF.iloc[split['val_indices']].reset_index(drop=True)  # 544 SV để chọn thiết lập
te42 = PRIMARY_DF.iloc[split['test_indices']].reset_index(drop=True)  # 545 SV TEST (chỉ dùng để chấm cuối; không dùng Y để tính ATE)
b0_dag42, _, _ = learn_b0_dag(tr42, va42, GLOBAL_CONFIG)  # học lại sơ đồ B0 của seed 42 (hai giá trị đi kèm bỏ qua bằng _)
b0_bn42 = fit_bn_on_dag(tr42, b0_dag42, GLOBAL_CONFIG)  # điền bảng BDeu trên TRAIN để có mạng Bayes B0 đầy đủ
h2_dag42, _, _ = select_h2_graph(run_artifacts[ref_seed]['H2_res'], tr42, va42, GLOBAL_CONFIG)  # sơ đồ H2 đã chọn ngưỡng tau của seed 42 (lấy từ kết quả mục 8.E)

print('Đang chạy refuter DoWhy trên DAG B0 và DAG H2 (seed 42, TRAIN)...')  # thông báo tiến trình
ref_b0 = dowhy_refute_on_graph(b0_dag42, tr42, GLOBAL_CONFIG, ref_seed)  # placebo + subset trên sơ đồ B0 (dict gồm original, placebo, subset, backdoor)
ref_h2 = dowhy_refute_on_graph(h2_dag42, tr42, GLOBAL_CONFIG, ref_seed)  # cùng hai refuter trên sơ đồ H2
print('Đang chạy kiểm định huấn luyện lại cho H1 (placebo + subset)...')  # thông báo tiến trình
ref_h1 = h1_retraining_refuter(b0_bn42, tr42, va42, te42, GLOBAL_CONFIG, ref_seed)  # H1 không dùng DoWhy được nên huấn luyện lại MLP với T xáo / 80% TRAIN
h1_orig = float(df_interv_raw[(df_interv_raw['Model'] == 'H1') & (df_interv_raw['Seed'] == ref_seed)]['ATE'].iloc[0])  # ATE gốc của H1 seed 42 (lấy từ Table B để so với refuter)

refuter_map = {  # gom kết quả của ba mô hình để lặp ở bước sau
    'B0': {'Refuter_Target': f"DoWhy LR on B0 DAG, backdoor {{{', '.join(ref_b0['backdoor'])}}}",  # tên refuter + tập điều chỉnh của B0
           'Original': ref_b0['original'], 'Placebo': ref_b0['placebo'], 'Subset': ref_b0['subset'],  # ATE gốc, ATE sau placebo, ATE sau subset
           'Sims': GLOBAL_CONFIG.placebo_simulations},  # số lần mô phỏng (30)
    'H1': {'Refuter_Target': 'Custom retraining of H1 MLP (permuted T / 80% TRAIN), ATE on C_test',  # H1 dùng refuter tự viết, không phải DoWhy
           'Original': h1_orig, 'Placebo': ref_h1['placebo'], 'Subset': ref_h1['subset'],
           'Sims': GLOBAL_CONFIG.h1_refute_sims},  # số lần huấn luyện lại (3)
    'H2': {'Refuter_Target': f"DoWhy LR on H2 DAG, backdoor {{{', '.join(ref_h2['backdoor'])}}}",  # tương tự B0 nhưng trên sơ đồ H2
           'Original': ref_h2['original'], 'Placebo': ref_h2['placebo'], 'Subset': ref_h2['subset'],
           'Sims': GLOBAL_CONFIG.placebo_simulations},
}

table_c_rows = []  # mỗi mô hình một dòng của Table C
for m in ['B0', 'H1', 'H2']:  # lặp qua ba mô hình
    sub_p = df_pred_raw[df_pred_raw['Model'] == m]  # các dòng dự báo (Table A) của mô hình m, 5 seed
    sub_i = df_interv_raw[df_interv_raw['Model'] == m]  # các dòng can thiệp (Table B) của mô hình m, 5 seed
    r = refuter_map[m]  # kết quả refuter của mô hình m
    table_c_rows.append({
        'Model': m, 'Refuter_Target': r['Refuter_Target'], 'Refuter_Seed': ref_seed, 'Simulations': r['Sims'],
        'Original_Effect': f"{r['Original']:+.4f}",  # ATE gốc, có dấu +/-
        'Placebo_Effect': f"{r['Placebo']:+.4f} (Pass: {abs(r['Placebo']) < 0.05})",  # đạt nếu |ATE placebo| < 0.05 (gần 0)
        'Subset_Effect': f"{r['Subset']:+.4f} (Pass: {abs(r['Subset'] - r['Original']) < 0.05})",  # đạt nếu ATE subset lệch gốc < 0.05
        'ATE_Mean_±_Std_5_Runs': mean_std(sub_i['ATE'], '+.4f'),  # ATE trung bình ± độ lệch chuẩn qua 5 seed
        'Paired_Spread_Max_Min': f"{sub_i['ATE'].max() - sub_i['ATE'].min():.4f}",  # độ tản: max trừ min qua 5 seed
        'Delta_ATE_vs_B0': mean_std(sub_i['Delta_ATE_vs_B0'], '+.4f') if m != 'B0' else '---',  # chênh so với B0 cùng seed (B0 để ---)
        'Fit_Time_mean_s': f"{sub_p['Fit_Time_s'].mean():.2f}", 'Query_Time_mean_s': f"{sub_p['Query_Time_s'].mean():.4f}",  # thời gian khớp và truy vấn trung bình
        'Hardware': HARDWARE,  # cấu hình máy để tái lập
    })
df_table_c = pd.DataFrame(table_c_rows)  # chuyển danh sách dòng thành bảng
df_table_c.to_csv(Path(GLOBAL_CONFIG.tables_dir) / 'Table_C_Robustness_Cost.csv', index=False)  # ghi file CSV Table C
print('[Xuất bản] Table C đã lưu.')  # xác nhận đã ghi file
'''
