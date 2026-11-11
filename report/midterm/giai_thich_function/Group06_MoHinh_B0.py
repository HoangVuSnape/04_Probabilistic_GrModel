# -*- coding: utf-8 -*-
"""
NHÓM 06 — Mô hình B0: học DAG truyền thống + CPD BDeu + truy vấn can thiệp (mục 8.B)

Ràng buộc thời gian, HillClimb với 3 hàm điểm chọn trên VALIDATION, khớp CPD BDeu, tính log-likelihood, và hàm
truy vấn q_m(t, C_i) dùng chung cho B0/H1/H2.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - build_b0_dag                               Cell 45 (def)
  - temporal_forbidden_edges                   Cell 45 (def)
  - fit_discrete_bayesian_model                Cell 45 (def)
  - fit_bn_on_dag                              Cell 45 (def)
  - validation_log_likelihood                  Cell 45 (def)
  - learn_b0_dag                               Cell 45 (def)
  - save_graph_artifacts                       Cell 45 (def)
  - query_model_effects                        Cell 45 (def)
"""


# ==============================================================================================================
# build_b0_dag  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dựng Domain DAG tham chiếu 7 nút, 11 cạnh (mỗi Z -> T, mỗi Z -> Y, T -> Y).
# LƯU Ý    : KHÔNG dùng để khớp B0. B0 học DAG từ dữ liệu; đồ thị này chỉ để so sánh.
# ==============================================================================================================
def build_b0_dag(cfg: Config) -> nx.DiGraph:
    """Domain DAG tham chiếu 7 nút (11 cạnh) — chỉ dùng để đối chiếu với DAG B0 học được, KHÔNG dùng để fit B0."""
    dag = nx.DiGraph()
    dag.add_nodes_from(cfg.all_nodes)
    for z in cfg.confounders:
        dag.add_edge(z, cfg.treatment)
        dag.add_edge(z, cfg.outcome)
    dag.add_edge(cfg.treatment, cfg.outcome)
    assert nx.is_directed_acyclic_graph(dag), 'Lỗi nghiêm trọng: Đồ thị B0 bị chu trình!'
    assert len(dag.edges()) == 11, f'Lỗi: Số cạnh Domain DAG phải là 11, thực tế có {len(dag.edges())}'
    return dag


# ==============================================================================================================
# temporal_forbidden_edges  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Danh sách cạnh cấm theo thời gian, dùng chung cho B0 (ExpertKnowledge) và H2 (mặt nạ M).
# CÁCH LÀM :
#   1. Cấm mọi cạnh đi ra từ Y: Y -> T, Y -> Z1..Z5 (kết cục không gây ra quá khứ).
#   2. Cấm mọi cạnh T -> Zk (học bổng không gây ra biến nền).
# LƯU Ý    : Cạnh giữa các Z với nhau KHÔNG bị cấm: chiều giữa các biến nền để thuật toán tự học.
# ❓ HỎI   : Có cấm Z -> Z không?
#   ➜ ĐÁP : Không. Các Z cùng tầng; chiều giữa chúng không ảnh hưởng truy vấn do(T) vì đều là tiền can thiệp.
# ==============================================================================================================
def temporal_forbidden_edges(cfg: Config) -> List[Tuple[str, str]]:
    """
    Danh sách cạnh cấm theo trật tự thời gian Z (nhập học) -> T (học bổng) -> Y (kết cục), dùng CHUNG cho B0 và H2:
    cấm Y -> (T, Z) và T -> Z.
    """
    cam = [(cfg.outcome, v) for v in cfg.all_nodes if v != cfg.outcome]  # cấm Y -> mọi nút khác (kết cục không gây ra quá khứ)
    cam += [(cfg.treatment, z) for z in cfg.confounders]  # cấm T -> Z1..Z5 (học bổng không gây ra biến nền)
    return cam  # danh sách cặp (nguồn, đích) bị cấm; cạnh Z -> Z KHÔNG bị cấm


# ==============================================================================================================
# fit_discrete_bayesian_model  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Khớp mọi CPD của một mạng Bayes rời rạc bằng ước lượng Bayes với tiên nghiệm BDeu (ESS mặc định 5).
# CÁCH LÀM :
#   1. Thử API mới của pgmpy (DiscreteBayesianEstimator); nếu lỗi thì dùng API cũ (BayesianEstimator).
#   2. model.check_model() để chắc mỗi CPD hợp lệ (tổng xác suất = 1, khớp cha).
# LÝ THUYẾT: Mỗi CPD là P(X | Pa(X)) = (đếm + ESS/(r q)) / (đếm cha + ESS/q): đếm tần suất có làm mịn. KHÔNG có
#            epoch hay gradient.
# ==============================================================================================================
def fit_discrete_bayesian_model(model: DiscreteBayesianNetwork, train_df: pd.DataFrame, ess: float = 5.0) -> DiscreteBayesianNetwork:
    """Khớp tham số mạng Bayes rời rạc với BDeu smoothing tương thích đa phiên bản pgmpy."""
    try:
        from pgmpy.parameter_estimator import DiscreteBayesianEstimator
        estimator = DiscreteBayesianEstimator(prior_type='BDeu', equivalent_sample_size=ess)
        model.fit(train_df, estimator=estimator)
    except Exception:
        from pgmpy.estimators import BayesianEstimator
        model.fit(train_df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=ess)
    assert model.check_model(), 'Lỗi: check_model() thất bại!'
    return model


# ==============================================================================================================
# fit_bn_on_dag  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Từ một DAG (networkx) tạo DiscreteBayesianNetwork đủ 7 nút rồi khớp CPD BDeu trên TRAIN.
# LƯU Ý    : Dùng cho B0 và cả H2: hai mô hình chỉ khác ĐỒ THỊ, cùng họ CPD và cùng ESS, nên so sánh công bằng.
# ==============================================================================================================
def fit_bn_on_dag(train_df: pd.DataFrame, dag: nx.DiGraph, cfg: Config) -> DiscreteBayesianNetwork:
    """Khớp CPD BDeu (cùng họ ước lượng và cùng ESS cho B0 và H2) trên một DAG cho trước."""
    model = DiscreteBayesianNetwork(list(dag.edges()))
    model.add_nodes_from(cfg.all_nodes)
    return fit_discrete_bayesian_model(model, train_df[list(cfg.all_nodes)], ess=cfg.b0_ess)


# ==============================================================================================================
# validation_log_likelihood  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Log-likelihood của toàn mạng trên một tập dữ liệu: sum_i sum_v log P(x_v | pa(x_v)). Dùng để CHỌN
#            trên VALIDATION.
# CÁCH LÀM :
#   1. Với mỗi nút, lấy CPD.
#   2. Đổi giá trị của nút và các cha thành chỉ số trong bảng CPD (searchsorted theo state_names).
#   3. Tra xác suất của từng dòng, lấy log, cộng dồn.
# ❓ HỎI   : Vì sao chọn hàm điểm bằng log-likelihood VAL chứ không phải bằng chính hàm điểm?
#   ➜ ĐÁP : Ba hàm điểm có thang khác nhau nên không so trực tiếp được; log-likelihood trên dữ liệu chưa thấy là
#           thước đo chung, khách quan.
# ==============================================================================================================
def validation_log_likelihood(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> float:
    """Log-likelihood của toàn mạng trên một tập dữ liệu: sum_i sum_v log P(x_v | pa(x_v))."""
    tong = 0.0  # tổng log-xác suất, càng gần 0 (ít âm) càng tốt
    for nut in model.nodes():  # duyệt từng nút của mạng (7 nút)
        cpd = model.get_cpds(nut)  # bảng xác suất của nút: P(nút | cha của nút)
        chi_so = tuple(np.searchsorted(np.array(cpd.state_names[v]), df[v].values) for v in cpd.variables)  # đổi giá trị của nút + cha thành chỉ số ô trong bảng, cho mọi dòng
        tong += float(np.log(cpd.values[chi_so]).sum())  # tra xác suất từng dòng, lấy log, cộng dồn
    return tong  # = log P(cả tập dữ liệu | mô hình)


# ==============================================================================================================
# learn_b0_dag  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tìm sơ đồ mũi tên (DAG) cho B0 từ dữ liệu TRAIN. Vì có nhiều cách "chấm điểm" sơ đồ cho ra kết quả
#            khác nhau, hàm thử 3 cách rồi để dữ liệu chưa thấy (VALIDATION) quyết định cách nào đúng hơn.
# ĐẦU VÀO  : train_df (2,541 SV), val_df (544 SV), cfg.
# ĐẦU RA   : (DAG được chọn, tên hàm điểm thắng, bảng 3 ứng viên để báo cáo).
# CÁCH LÀM :
#   1. Khai báo luật bắt buộc trước khi học (ExpertKnowledge): CẤM mọi mũi tên đi ngược thời gian (Y -> bất kỳ,
#      T -> Z) và BẮT BUỘC có T -> Y. Vì sao: dữ liệu quan sát chỉ cho biết hai biến "đi cùng nhau", không cho
#      biết chiều; không cấm thì thuật toán có thể vẽ "tốt nghiệp gây ra điểm đầu vào" (đã thấy ở mục 3).
#   2. Với từng "giám khảo" BIC, BDeu (ESS=5), K2: chạy HillClimb — bắt đầu từ sơ đồ trống, thử thêm / bớt / đảo
#      một mũi tên, giữ thay đổi làm điểm tăng nhiều nhất, lặp đến khi không cải thiện. Kết quả: 3 sơ đồ, khác
#      nhau chủ yếu ở số mũi tên.
#   3. Vì sao 3 giám khảo cho 3 sơ đồ khác nhau: chúng khác nhau ở mức "khó tính" với mũi tên mới. BIC phạt mỗi
#      tham số thêm rất nặng (theo log N); BDeu với ESS nhỏ cũng khá khắt khe; K2 dễ dãi hơn nên giữ thêm mũi
#      tên. Không có lý thuyết nào nói trước giám khảo nào đúng cho dữ liệu này.
#   4. Để dữ liệu tự phân xử: với mỗi sơ đồ, điền bảng CPD trên TRAIN rồi tính log-likelihood trên VALIDATION —
#      tức "sơ đồ này đoán dữ liệu CHƯA TỪNG THẤY tốt cỡ nào". Không so trực tiếp điểm BIC với điểm K2 được vì
#      chúng ở hai thang đo khác nhau; log-likelihood trên VAL là thước đo chung, khách quan.
#   5. Chọn sơ đồ có log-likelihood VAL cao nhất (ít âm nhất), rồi đóng băng. Kết quả: K2 thắng ở cả 5 seed.
#   6. assert: sơ đồ không có vòng tròn và không chứa mũi tên bị cấm — lưới an toàn, nếu sai thì dừng ngay chứ
#      không chạy tiếp với sơ đồ hỏng.
# LÝ THUYẾT: Số liệu thật cho thấy vì sao K2 thắng: ở seed 42, BIC và BDeu cùng ra 9 mũi tên với Pa(Y) = {T, học
#            phí, tuổi}, log-likelihood VAL = -2211.11; K2 ra 11 mũi tên, giữ thêm Debtor -> Y, log-likelihood
#            VAL = -2208.37 (tốt hơn 2.74). Quy luật lặp lại ở cả 5 seed: giám khảo khắt khe bỏ Debtor -> Y, K2
#            giữ lại, và lần nào sơ đồ có Debtor -> Y cũng đoán VAL tốt hơn (seed 46: -2174.84 so với -2182.63;
#            seed 44 BDeu cũng giữ Debtor và suýt bằng K2). => Debtor thực sự giúp dự đoán tốt nghiệp trên dữ
#            liệu mới; BIC đã "cắt quá tay". Đây là lý do chọn bằng VAL thay vì chọn cố định một giám khảo từ
#            đầu.
# LƯU Ý    : T -> Y bắt buộc KHÔNG ép học bổng phải có tác dụng: nó chỉ bảo đảm sơ đồ có chỗ để đo. Độ lớn tác
#            dụng vẫn do bảng CPD học từ dữ liệu quyết định; nếu học bổng vô dụng, hai cột "có / không học bổng"
#            trong bảng sẽ gần bằng nhau và ATE ≈ 0. Bằng chứng: H2 không bị ép mà vẫn tự học ra T -> Y ở cả 5
#            seed.
# ❓ HỎI   : Vì sao không dùng luôn BIC cho đơn giản?
#   ➜ ĐÁP : Vì BIC bỏ mất Debtor -> Y, và trên dữ liệu chưa thấy (VAL) sơ đồ đó đoán kém hơn ở cả 5 seed. Chọn
#           bằng VAL là để dữ liệu quyết định, không phải nhóm chọn theo ý thích.
# ❓ HỎI   : Chọn trên VAL có làm rò rỉ TEST không?
#   ➜ ĐÁP : Không. Chỉ dùng TRAIN để học và VAL để chọn; TEST chỉ dùng một lần ở cuối cho Table A, B.
# ❓ HỎI   : B0 có "train" không?
#   ➜ ĐÁP : Có, nhưng không bằng gradient: tìm sơ đồ bằng leo đồi, điền bảng bằng đếm có làm mịn BDeu; lựa chọn
#           duy nhất là giám khảo, chọn trên VAL.
# ❓ HỎI   : Sao phải ép T -> Y? Như vậy là áp đặt kết quả à?
#   ➜ ĐÁP : Không áp đặt độ lớn: cạnh bắt buộc chỉ để câu hỏi do(T) có nghĩa, còn tác dụng lớn hay nhỏ vẫn do dữ
#           liệu. H2 không bị ép mà vẫn tự học ra cạnh này.
# ==============================================================================================================
def learn_b0_dag(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config) -> Tuple[nx.DiGraph, str, List[Dict[str, Any]]]:
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


# ==============================================================================================================
# save_graph_artifacts  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Lưu danh sách nút, cạnh và thông tin phụ (hàm điểm, tau, log H2...) ra artifacts/graphs/<tên>.json.
# LƯU Ý    : File JSON này là nơi notebook xem log (training_log_viewer.ipynb) đọc lại log H2.
# ==============================================================================================================
def save_graph_artifacts(dag: nx.DiGraph, name: str, cfg: Config, extra: Optional[Dict[str, Any]] = None) -> None:
    """Lưu danh sách cạnh (và thông tin phụ) của một DAG ra artifacts/graphs/<name>.json (Slide 44, 76)."""
    noi_dung = {'nodes': list(cfg.all_nodes), 'edges': [list(e) for e in dag.edges()]}
    if extra:
        noi_dung.update(extra)
    with open(Path(cfg.graphs_dir) / f'{name}.json', 'w', encoding='utf-8') as tep:
        json.dump(noi_dung, tep, indent=2, ensure_ascii=False, default=str)


# ==============================================================================================================
# query_model_effects  (def)
# Vị trí: Cell 45 · mục 8.B Mô hình B0 — học DAG truyền thống + CPD BDeu
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hàm truy vấn CHUNG cho B0/H1/H2: với mỗi hồ sơ C_i tính q(0, C_i), q(1, C_i) và xác suất dự báo
#            P(Y=1 | T thật, Pa(Y)).
# ĐẦU VÀO  : model (mạng Bayes đã khớp), df (các hồ sơ, ví dụ C_test), cfg.
# ĐẦU RA   : (q0, q1, pred) — mỗi cái là mảng dài bằng số hồ sơ.
# CÁCH LÀM :
#   1. Lấy CPD của Y; cha của Y là Pa(Y).
#   2. Nếu Y không có cha: mọi hồ sơ cùng một xác suất.
#   3. Tạo chỉ số tra bảng: chiều đầu = 1 (lấy P(Y=1)); với cha là T thì đặt 0 (cho q0), 1 (cho q1), hoặc T thật
#      (cho pred); cha khác lấy giá trị của hồ sơ.
#   4. Tra bảng CPD bằng numpy advanced indexing cho cả 545 hồ sơ cùng lúc (khoảng 0.2 ms).
# LÝ THUYẾT: q_m(t, C_i) = P_m(Y=1 | T=t, Pa(Y)\T = C_i). Vì mặt nạ cấm T -> Z, mọi cha khác của Y là tiền can
#            thiệp nên là tập điều chỉnh cửa sau hợp lệ, do đó q_m(t, C_i) = P(Y=1 | do(T=t), C_i). risk_m(t) =
#            trung bình q_m(t, C_i) trên C_test; ATE = risk(1) - risk(0). Không dùng Y_test.
# ❓ HỎI   : Sao không dùng Variable Elimination của pgmpy?
#   ➜ ĐÁP : Kết quả như nhau vì chỉ cần CPD của Y; tra bảng trực tiếp nhanh và minh bạch hơn (mục 8.2c tính tay
#           lại để đối chiếu).
# ==============================================================================================================
def query_model_effects(model: DiscreteBayesianNetwork, df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
