# -*- coding: utf-8 -*-
"""
NHÓM 07 — Mô hình H1: thay CPD của Y bằng mạng nơ-ron trên DAG của B0 (mục 8.C)

MLP PyTorch ước lượng P(Y=1 | Pa(Y)), huấn luyện có early stopping, lưu checkpoint, rồi chuyển MLP thành bảng
CPD để gắn lại vào mạng Bayes.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - NeuralCPDModule                            Cell 47 (class)
  - encode_h1_features                         Cell 47 (def)
  - h1_input_dim                               Cell 47 (def)
  - train_h1_mlp                               Cell 47 (def)
  - save_h1_checkpoint                         Cell 47 (def)
  - build_h1_bayesian_network                  Cell 47 (def)
"""


# ==============================================================================================================
# NeuralCPDModule  (class)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mạng MLP ước lượng P(Y=1 | Pa(Y)) cho H1. Đầu ra là logit (chưa qua sigmoid).
# CÁCH LÀM :
#   1. Linear(số chiều vào -> 32) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(32 -> 16) -> ReLU -> Linear(16
#      -> 1).
#   2. forward trả logit; xác suất = sigmoid(logit).
# LÝ THUYẾT: Với Pa(Y) = {T, Z2, Z3, Z5} (4 biến nhị phân) thì số tham số = (4x32+32) + 64 (BatchNorm) +
#            (32x16+16) + (16+1) = 769.
# ❓ HỎI   : Vì sao cần BatchNorm và Dropout?
#   ➜ ĐÁP : BatchNorm ổn định huấn luyện; Dropout và weight decay chống overfit. Thực tế đầu vào chỉ 16 tổ hợp
#           nên mạng hội tụ rất nhanh.
# ==============================================================================================================
class NeuralCPDModule(nn.Module):
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


# ==============================================================================================================
# encode_h1_features  (def)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mã hóa ĐÚNG các nút cha của Y làm đầu vào MLP (Slide 60): biến 3 trạng thái -> one-hot 3 chiều,
#            biến nhị phân giữ nguyên.
# LƯU Ý    : STATE_CARD là bảng số trạng thái của từng nút (Z1 có 3, còn lại 2).
# ==============================================================================================================
def encode_h1_features(df: pd.DataFrame, parents: List[str]) -> torch.Tensor:
    """Mã hóa DUY NHẤT các nút cha của Y (Slide 60): biến 3 trạng thái -> one-hot, biến nhị phân giữ nguyên."""
    khoi: List[np.ndarray] = []  # mỗi cha một khối cột
    for p in parents:
        if STATE_CARD[p] > 2:  # biến có 3 trạng thái (Z1 điểm vào: 0/1/2)
            khoi.append(np.eye(STATE_CARD[p])[df[p].values.astype(int)])  # one-hot: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
        else:  # biến nhị phân 0/1: giữ nguyên một cột
            khoi.append(df[[p]].values.astype(np.float32))
    return torch.tensor(np.hstack(khoi).astype(np.float32), dtype=torch.float32)  # ghép các khối thành ma trận đầu vào cho MLP


# ==============================================================================================================
# h1_input_dim  (def)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Số chiều đầu vào của MLP: cộng 3 cho mỗi biến 3 trạng thái, 1 cho mỗi biến nhị phân.
# ==============================================================================================================
def h1_input_dim(parents: List[str]) -> int:
    """Số chiều đầu vào của MLP ứng với tập cha."""
    return int(sum(STATE_CARD[p] if STATE_CARD[p] > 2 else 1 for p in parents))


# ==============================================================================================================
# train_h1_mlp  (def)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Huấn luyện MLP của H1 trên TRAIN bằng BCE, theo dõi loss VAL sau mỗi epoch; mặc định có early
#            stopping.
# ĐẦU VÀO  : train_df, val_df, parents = Pa(Y) từ DAG B0, cfg, seed, dung_som (True mặc định).
# ĐẦU RA   : (model, nhat_ky) — nhật ký gồm train_loss, val_loss từng epoch, best_epoch, epochs_run, số tham số,
#            kiến trúc.
# CÁCH LÀM :
#   1. set_seed; mã hóa X_train, X_val; DataLoader batch 64, xáo trộn.
#   2. Loss = BCEWithLogitsLoss (sigmoid + binary cross-entropy, ổn định số học); Adam lr 0.005, weight decay
#      1e-4.
#   3. Mỗi epoch: duyệt các batch, lan truyền ngược, cập nhật; cộng dồn loss TRAIN (dropout đang bật).
#   4. Cuối epoch: model.eval(), tính loss VAL (không dropout).
#   5. Nếu loss VAL tốt hơn kỷ lục: lưu bản sao trọng số, đặt lại bộ đếm; ngược lại tăng bộ đếm.
#   6. dung_som=True: bộ đếm đạt 15 thì dừng và khôi phục trọng số tốt nhất. dung_som=False: chạy đủ 150 epoch,
#      giữ trọng số epoch cuối.
# ❓ HỎI   : Bao nhiêu epoch?
#   ➜ ĐÁP : Tối đa 150; thực tế dừng sau 17–39 epoch, tốt nhất ở epoch 2–4 (seed 43 là 24) vì đầu vào chỉ có 16
#           tổ hợp.
# ❓ HỎI   : Loss TRAIN cao hơn VAL là sao?
#   ➜ ĐÁP : Loss TRAIN đo khi dropout bật nên bị đội lên; loss VAL đo ở chế độ eval.
# ==============================================================================================================
def train_h1_mlp(train_df: pd.DataFrame, val_df: pd.DataFrame, parents: List[str], cfg: Config,
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


# ==============================================================================================================
# save_h1_checkpoint  (def)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Lưu trọng số và nhật ký H1 ra artifacts/checkpoints/h1_seed<seed><hậu tố>.pt.
# LƯU Ý    : Hậu tố "_early_stop" / "_no_early_stop" dùng ở mục 8.E2 để lưu hai bản riêng.
# ==============================================================================================================
def save_h1_checkpoint(model: NeuralCPDModule, nhat_ky: Dict[str, Any], seed: int, cfg: Config,
                       hau_to: str = '') -> Path:
    """Lưu trọng số + cấu hình H1 (Slide 44, 76) ra artifacts/checkpoints/h1_seed<seed><hau_to>.pt."""
    duong_dan = Path(cfg.checkpoints_dir) / f'h1_seed{seed}{hau_to}.pt'
    torch.save({'state_dict': model.state_dict(), 'meta': nhat_ky}, duong_dan)
    return duong_dan


# ==============================================================================================================
# build_h1_bayesian_network  (def)
# Vị trí: Cell 47 · mục 8.C Mô hình H1 — CPD nơ-ron trên DAG B0
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Biến MLP thành bảng CPD của Y rồi gắn vào bản sao của B0: H1 = B0 với CPD(Y) do mạng nơ-ron sinh
#            ra.
# CÁCH LÀM :
#   1. Liệt kê MỌI cấu hình cha của Y (itertools.product các trạng thái).
#   2. Cho MLP dự đoán P(Y=1) ở từng cấu hình; chặn trong [1e-4, 1 - 1e-4] để tránh log 0.
#   3. Tạo TabularCPD với 2 hàng [1-p, p].
#   4. Dựng mạng mới cùng cạnh với B0, chép mọi CPD khác của B0, thay CPD của Y bằng CPD mới; check_model.
# LÝ THUYẾT: Vì H1 vẫn là một mạng Bayes trên cùng DAG, truy vấn can thiệp dùng y hệt query_model_effects như
#            B0: so sánh công bằng, chỉ khác cách tham số hóa CPD(Y).
# ❓ HỎI   : H1 khác B0 ở đâu?
#   ➜ ĐÁP : Cùng DAG, cùng CPD các nút khác; chỉ CPD của Y được học bằng MLP thay vì đếm BDeu.
# ==============================================================================================================
def build_h1_bayesian_network(b0_model: DiscreteBayesianNetwork, h1_mlp: NeuralCPDModule,
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
