# -*- coding: utf-8 -*-
"""
NHÓM 08 — Mô hình H2: học DAG bằng NOTEARS-MLP có mặt nạ M (mục 8.D)

Mặt nạ cạnh cấm, mạng NOTEARS phi tuyến, tối ưu augmented Lagrangian, lọc ngưỡng tau thành DAG và chọn tau
trên VALIDATION.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - build_domain_mask                          Cell 49 (def)
  - NotearsMLP                                 Cell 49 (class)
  - train_notears_mlp                          Cell 49 (def)
  - graph_from_weights                         Cell 49 (def)
  - select_h2_graph                            Cell 49 (def)
"""


# ==============================================================================================================
# build_domain_mask  (def)
# Vị trí: Cell 49 · mục 8.D Mô hình H2 — NOTEARS-MLP với mặt nạ M
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tạo mặt nạ M (7x7) cho H2: M[i, j] = 0 nếu cạnh i -> j bị cấm (cùng danh sách với B0) và cấm tự
#            nối.
# LÝ THUYẾT: Slide 63: H2 phải dùng cùng ràng buộc miền với B0. Nhân M vào trọng số làm cạnh cấm luôn bằng 0.
# ==============================================================================================================
def build_domain_mask(cfg: Config) -> np.ndarray:
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


# ==============================================================================================================
# NotearsMLP  (class)
# Vị trí: Cell 49 · mục 8.D Mô hình H2 — NOTEARS-MLP với mặt nạ M
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mạng NOTEARS phi tuyến: mỗi nút j có một MLP riêng f_j dự đoán X_j từ các nút khác; độ mạnh cạnh i
#            -> j suy ra từ trọng số tầng 1.
# CÁCH LÀM :
#   1. fc1 có kích thước [nút j, nơ-ron ẩn, đầu vào i]; masked_fc1 nhân mặt nạ M nên đầu vào bị cấm có trọng số
#      và gradient bằng 0.
#   2. forward: h = sigmoid(fc1 . X + b1) cho từng nút, rồi X_hat_j = fc2 . h + b2.
#   3. adjacency_sq: W[i, j]^2 = tổng bình phương trọng số tầng 1 từ đầu vào i vào MLP của nút j.
# LÝ THUYẾT: Zheng et al. (2020, NOTEARS-MLP): W[i, j] = chuẩn L2 của cột trọng số tầng 1; nếu W[i, j] = 0 thì
#            f_j không phụ thuộc X_i, tức không có cạnh i -> j.
# ==============================================================================================================
class NotearsMLP(nn.Module):
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


# ==============================================================================================================
# train_notears_mlp  (def)
# Vị trí: Cell 49 · mục 8.D Mô hình H2 — NOTEARS-MLP với mặt nạ M
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tối ưu NOTEARS-MLP bằng augmented Lagrangian để vừa khớp dữ liệu vừa ép đồ thị không có chu trình.
# ĐẦU VÀO  : train_df (7 biến), cfg, seed, lambda_l1.
# ĐẦU RA   : Dict: W, mask, h_final, converged, outer_iters, h_history, recon_history, forbidden_weight_max,
#            lambda_l1.
# CÁCH LÀM :
#   1. Chuẩn hóa X (trừ trung bình, chia độ lệch chuẩn).
#   2. Hàm mục tiêu mỗi bước: recon + lambda_1 x ||W||_1 + alpha x h + (rho/2) x h^2, với recon = MSE/2 tái tạo
#      X và h = tr(exp(W∘W)) - d.
#   3. Vòng TRONG: 300 bước Adam (lr 0.01).
#   4. Sau vòng trong: nếu h không giảm xuống dưới 1/4 lần trước thì tăng rho x 10 và chạy lại vòng trong.
#   5. Vòng NGOÀI: cập nhật alpha += rho x h; ghi h và recon vào lịch sử; dừng khi h < 1e-6 hoặc hết 12 vòng.
#   6. Trả W = căn của adjacency_sq và kiểm tra trọng số lớn nhất ở ô bị cấm (phải bằng 0).
# LÝ THUYẾT: h(W) = 0 khi và chỉ khi đồ thị không có chu trình. Augmented Lagrangian biến ràng buộc cứng h = 0
#            thành chuỗi bài toán có phạt ngày càng nặng.
# ❓ HỎI   : H2 có epoch không?
#   ➜ ĐÁP : Không. Mỗi vòng ngoài gồm 300 bước Adam; cả 5 seed hội tụ sau 7–8 vòng với h khoảng 1e-7.
# ❓ HỎI   : Vì sao sai số tái tạo tăng qua các vòng?
#   ➜ ĐÁP : Phạt chu trình càng nặng thì mô hình càng phải bỏ cạnh tạo vòng, chấp nhận khớp kém hơn: đánh đổi
#           giữa khớp dữ liệu và là DAG hợp lệ.
# ==============================================================================================================
def train_notears_mlp(train_df: pd.DataFrame, cfg: Config, seed: int, lambda_l1: float) -> Dict[str, Any]:
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


# ==============================================================================================================
# graph_from_weights  (def)
# Vị trí: Cell 49 · mục 8.D Mô hình H2 — NOTEARS-MLP với mặt nạ M
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Chuyển ma trận W thành DAG: giữ cạnh có |W| >= tau, cắt chu trình còn sót, rồi thêm cạnh bắt buộc T
#            -> Y nếu thiếu.
# CÁCH LÀM :
#   1. Thêm mọi cạnh i -> j có W[i, j] >= tau.
#   2. Trong khi còn chu trình: tìm một chu trình và bỏ cạnh có trọng số nhỏ nhất trong đó (luật sửa khai báo
#      trước, Slide 79).
#   3. Nếu thiếu T -> Y thì thêm và bật cờ "forced".
# LƯU Ý    : Thực tế chưa seed nào phải cắt chu trình hay ép T -> Y ở tau được chọn.
# ==============================================================================================================
def graph_from_weights(W: np.ndarray, threshold: float, cfg: Config) -> Tuple[nx.DiGraph, List[Tuple[str, str]], bool]:
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


# ==============================================================================================================
# select_h2_graph  (def)
# Vị trí: Cell 49 · mục 8.D Mô hình H2 — NOTEARS-MLP với mặt nạ M
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Chọn ngưỡng tau của H2 trên VALIDATION: với mỗi tau, khớp CPD BDeu trên TRAIN và tính
#            log-likelihood VAL, giữ tau tốt nhất.
# LƯU Ý    : Bảng tau_table được lưu vào JSON để xem lại; tau thường là 0.1 hoặc 0.2.
# ==============================================================================================================
def select_h2_graph(res: Dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame,
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
