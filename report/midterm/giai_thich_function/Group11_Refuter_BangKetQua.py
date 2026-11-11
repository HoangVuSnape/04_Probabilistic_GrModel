# -*- coding: utf-8 -*-
"""
NHÓM 11 — Refuter (Table C), hiển thị bảng, positivity, đọc lại đồ thị và q_m minh bạch (mục 8.F, 8.2b, 8.2c)

Refuter DoWhy trên DAG của B0/H2, kiểm định huấn luyện lại cho H1, đọc và hiển thị các bảng, bảng 4 ô học bổng
x học phí, đọc DAG đã lưu cho Hình 8.1 và tính tay q_m(t, C_i).

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - dowhy_refute_on_graph                      Cell 57 (def)
  - h1_retraining_refuter                      Cell 57 (def)
  - doan_ma_cell_57_chay_refuter_va_table_c    Cell 57 (đoạn mã)
  - doc_bang_csv                               Cell 58 (def)
  - bang_hien_thi                              Cell 58 (def)
  - bang_positivity_T1_T2                      Cell 59 (def)
  - doc_do_thi_da_luu                          Cell 61 (def)
  - mat_na_thoi_gian                           Cell 61 (def)
  - bang_tom_tat_do_thi                        Cell 61 (def)
  - ma_hoa_7_bien                              Cell 62 (def)
  - tai_mlp_h1                                 Cell 62 (def)
  - ma_hoa_dau_vao_h1                          Cell 62 (def)
  - q_b0                                       Cell 62 (def)
  - q_h1                                       Cell 62 (def)
"""


# ==============================================================================================================
# dowhy_refute_on_graph  (def)
# Vị trí: Cell 57 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hai refuter DoWhy (Slide 69–71) trên ĐÚNG DAG của B0 hoặc H2, dùng ước lượng hỗ trợ hồi quy tuyến
#            tính, cho Table C.
# CÁCH LÀM :
#   1. CausalModel với DAG của mô hình; identify_effect; estimate_effect backdoor.linear_regression.
#   2. Placebo: hoán vị T (placebo_type="permute"), 30 lần -> hiệu ứng phải về khoảng 0.
#   3. Subset: lấy ngẫu nhiên 80% dữ liệu, 30 lần -> hiệu ứng phải gần gốc.
# LƯU Ý    : Refuter này kiểm tra (đồ thị + ước lượng tuyến tính), không kiểm tra trực tiếp CPD BDeu hay MLP.
# ==============================================================================================================
def dowhy_refute_on_graph(dag: nx.DiGraph, df: pd.DataFrame, cfg: Config, seed: int) -> Dict[str, Any]:
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


# ==============================================================================================================
# h1_retraining_refuter  (def)
# Vị trí: Cell 57 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Kiểm định tương tự refuter cho H1 nơ-ron (vì refuter tuyến tính không áp dụng được): huấn luyện lại
#            MLP trong 3 lần.
# CÁCH LÀM :
#   1. Placebo: hoán vị T trong TRAIN và VAL, huấn luyện lại MLP, tính ATE trên C_test -> kỳ vọng khoảng 0 (thực
#      tế +0.018).
#   2. Subset: huấn luyện lại trên 80% TRAIN, tính ATE trên C_test -> kỳ vọng gần gốc (+0.237 so với +0.245).
#   3. Lấy trung bình 3 lần.
# LƯU Ý    : Được dán nhãn riêng trong Table C ("Custom retraining") để không nhầm với refuter DoWhy.
# ==============================================================================================================
def h1_retraining_refuter(b0_model: DiscreteBayesianNetwork, train_df: pd.DataFrame, val_df: pd.DataFrame,
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


# ==============================================================================================================
# doan_ma_cell_57_chay_refuter_va_table_c  (đoạn mã tuần tự)
# Vị trí: Cell 57 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Chạy 4 việc cho seed 42 rồi xuất Table C: (1) lấy sơ đồ B0, H2 và mạng B0 của seed 42; (2) refuter
#            DoWhy trên B0 và H2; (3) refuter huấn luyện lại cho H1; (4) gom thành bảng Table C (độ bền vững +
#            chi phí).
# CÁCH LÀM :
#   1. Lấy đúng TRAIN, VAL, TEST của seed 42 từ split đã đóng băng.
#   2. Học lại sơ đồ B0 và lấy sơ đồ H2 đã chọn; điền bảng BDeu để có mạng Bayes B0.
#   3. Hai refuter DoWhy (placebo và subset, mỗi loại 30 lần) trên sơ đồ B0 và H2.
#   4. Refuter tự viết cho H1 (huấn luyện lại 3 lần).
#   5. Gom kết quả ba mô hình thành một dòng mỗi mô hình, rồi ghi Table C ra CSV.
# LÝ THUYẾT: Refuter chỉ kiểm tra độ ổn định của ước lượng; không chứng minh sơ đồ đúng hay không có nhiễu ẩn.
#            Y_test không được dùng để tính ATE.
# ==============================================================================================================
def doan_ma_cell_57_chay_refuter_va_table_c():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    ref_seed = 42  # seed tham chiếu cho Table C (seed đầu tiên của 5 run)
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



# ==============================================================================================================
# doc_bang_csv  (def)
# Vị trí: Cell 58 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Đọc một bảng kết quả từ CSV trong artifacts/tables/.
# ==============================================================================================================
def doc_bang_csv(duong_dan_tep: Path) -> pd.DataFrame:
    """
    Hàm này để làm gì: Đọc bảng dữ liệu đối chuẩn đã lưu từ tệp CSV.

    Nhận vào:
        duong_dan_tep (Path): Đường dẫn tệp CSV bảng kết quả.

    Trả ra:
        pd.DataFrame: Bảng kết quả định dạng pandas.
    """
    return pd.read_csv(duong_dan_tep)


# ==============================================================================================================
# bang_hien_thi  (def)
# Vị trí: Cell 58 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tạo BẢN SAO bảng để hiển thị, làm tròn 4 chữ số các ô số nằm trong cột kiểu chữ (cột trộn số với
#            "mean ± std").
# LƯU Ý    : Dữ liệu gốc không đổi vì các cell sau (forest plot) còn dùng.
# ==============================================================================================================
def bang_hien_thi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Tạo BẢN SAO để hiển thị, trong đó các ô số nằm trong cột kiểu chuỗi (cột trộn số với
    chuỗi 'mean ± std') cũng được làm tròn 4 chữ số thập phân như các cột số. Dữ liệu gốc không đổi.

    Nhận vào:
        df (pd.DataFrame): Bảng đọc từ CSV.

    Trả ra:
        pd.DataFrame: Bảng đã định dạng để display().
    """
    def _o(v):
        if isinstance(v, str):
            try:
                return _dinh_dang_so(float(v))
            except ValueError:
                return v
        return v
    ban_sao = df.copy()
    for cot in ban_sao.columns:
        if not pd.api.types.is_numeric_dtype(ban_sao[cot]):
            ban_sao[cot] = ban_sao[cot].map(_o)
    return ban_sao


# ==============================================================================================================
# bang_positivity_T1_T2  (def)
# Vị trí: Cell 59 · mục 8.F Refuter và chi phí tính toán — Table C
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng 4 ô (học bổng x học phí đúng hạn) kèm tỷ lệ tốt nghiệp, để kiểm tra positivity của T3 và dấu
#            hiệu nhân quả ngược của T2.
# ❓ HỎI   : Bảng cho thấy gì?
#   ➜ ĐÁP : Ô có học bổng nhưng chưa đóng đúng hạn chỉ 37 SV (positivity yếu); nhóm không học bổng và chưa đóng
#           đúng hạn chỉ 4.5% tốt nghiệp (gợi ý bỏ học -> ngừng đóng phí).
# ==============================================================================================================
def bang_positivity_T1_T2(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Lập bảng 4 ô (Học bổng T1 x Học phí đúng hạn T2) kèm tỷ lệ tốt nghiệp,
    để kiểm tra điều kiện positivity của tương phản kép T3 và dấu hiệu nhân quả ngược của T2.

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        pd.DataFrame: Số sinh viên và tỷ lệ tốt nghiệp trong từng ô (T1, T2).
    """
    return (du_lieu.groupby(['Scholarship_holder', 'Tuition_fees_up_to_date'])['Target_Graduate']
            .agg(So_SV='size', Ty_le_tot_nghiep='mean').reset_index())


# ==============================================================================================================
# doc_do_thi_da_luu  (def)
# Vị trí: Cell 61 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Đọc file JSON đồ thị do vòng lặp 5 run đã lưu (artifacts/graphs/<tên>.json).
# ==============================================================================================================
def doc_do_thi_da_luu(ten: str) -> Dict[str, object]:
    """
    Hàm này để làm gì: Đọc file JSON đồ thị do pipeline mục 8 vừa lưu (artifacts/graphs/<ten>.json).

    Nhận vào:
        ten (str): Tên file không kèm đuôi, ví dụ 'b0_dag_seed42'.

    Trả ra:
        Dict[str, object]: Danh sách nút, cạnh và thông tin chọn mô hình.
    """
    with open(THU_MUC_HINH / f'{ten}.json', 'r', encoding='utf-8') as tep:
        return json.load(tep)


# ==============================================================================================================
# mat_na_thoi_gian  (def)
# Vị trí: Cell 61 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dựng lại mặt nạ M để vẽ heatmap Hình 8.1(a): cấm tự nối, cấm Y -> mọi nút, cấm T -> Z.
# LƯU Ý    : Cùng nội dung với build_domain_mask nhưng viết lại theo thứ tự nút của hình.
# ==============================================================================================================
def mat_na_thoi_gian(cac_nut: Tuple[str, ...]) -> np.ndarray:
    """
    Hàm này để làm gì: Tạo mặt nạ M (0 = cấm cạnh hàng -> cột): cấm tự nối, cấm Y -> (T, Z), cấm T -> Z.

    Nhận vào:
        cac_nut (Tuple[str, ...]): Thứ tự nút.

    Trả ra:
        np.ndarray: Ma trận mặt nạ.
    """
    m = np.ones((len(cac_nut), len(cac_nut)))
    np.fill_diagonal(m, 0)
    iy, it = cac_nut.index('Y'), cac_nut.index('T')
    m[iy, :] = 0
    for k, ten in enumerate(cac_nut):
        if ten.startswith('Z'):
            m[it, k] = 0
    return m


# ==============================================================================================================
# bang_tom_tat_do_thi  (def)
# Vị trí: Cell 61 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng tóm tắt đồ thị 5 seed: hàm điểm B0 được chọn, số cạnh, Pa(Y); tau, số cạnh, Pa(Y), h(W), hội
#            tụ, có ép T -> Y không của H2.
# ==============================================================================================================
def bang_tom_tat_do_thi(cac_seed: Tuple[int, ...]) -> pd.DataFrame:
    """
    Hàm này để làm gì: Tổng hợp lựa chọn mô hình theo seed: hàm điểm B0, số cạnh, Pa(Y); ngưỡng, hội tụ và
    cạnh T->Y có bị ép thêm hay không của H2.

    Nhận vào:
        cac_seed (Tuple[int, ...]): Các seed.

    Trả ra:
        pd.DataFrame: Một dòng mỗi seed.
    """
    dong: List[Dict[str, object]] = []
    for s in cac_seed:
        b0 = doc_do_thi_da_luu(f'b0_dag_seed{s}')
        h2 = doc_do_thi_da_luu(f'h2_dag_seed{s}')
        dong_tau = [t for t in h2['tau_table'] if t['tau'] == h2['tau_selected_on_val']][0]
        dong.append({'Seed': s, 'B0_score(val)': b0['score_selected_on_val'], 'B0_edges': len(b0['edges']),
                     'B0_Pa(Y)': sorted(u for u, v in b0['edges'] if v == 'Y'),
                     'H2_tau(val)': h2['tau_selected_on_val'], 'H2_edges': len(h2['edges']),
                     'H2_Pa(Y)': sorted(u for u, v in h2['edges'] if v == 'Y'),
                     'H2_h(W)': f"{h2['h_final']:.1e}", 'H2_converged': h2['converged'],
                     'H2_T->Y_forced': dong_tau['T->Y_forced'], 'H2_max|W|_forbidden': h2['forbidden_weight_max']})
    return pd.DataFrame(dong)


# ==============================================================================================================
# ma_hoa_7_bien  (def)
# Vị trí: Cell 62 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mã hóa lại 7 biến đúng như mục 8.A (từ dữ liệu cột gốc) để mục 8.2c dùng lại split và checkpoint đã
#            lưu.
# ==============================================================================================================
def ma_hoa_7_bien(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Mã hóa 7 biến đúng như mục 8.A để dùng lại DAG/checkpoint đã lưu.

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        pd.DataFrame: Cột Z1..Z5, T, Y (chỉ số 0..N-1, khớp file split).
    """
    d = du_lieu.reset_index(drop=True)
    return pd.DataFrame({
        'Z1': pd.cut(d['Admission_grade'], bins=[-np.inf, 120.0, 140.0, np.inf], labels=[0, 1, 2]).astype(int),
        'Z2': d['Debtor'].astype(int), 'Z3': d['Tuition_fees_up_to_date'].astype(int),
        'Z4': (d['Mothers_qualification'] >= 13).astype(int), 'Z5': (d['Age_at_enrollment'] > 20).astype(int),
        'T': d['Scholarship_holder'].astype(int), 'Y': d['Target_Graduate'].astype(int)})


# ==============================================================================================================
# tai_mlp_h1  (def)
# Vị trí: Cell 62 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dựng lại kiến trúc MLP của H1 và nạp trọng số từ checkpoint h1_seed42.pt (chế độ eval).
# LƯU Ý    : Tạo Module().net để khớp tên khóa "net.*" trong state_dict đã lưu.
# ==============================================================================================================
def tai_mlp_h1(duong_dan: Path) -> Tuple[torch.nn.Module, List[str]]:
    """
    Hàm này để làm gì: Dựng lại kiến trúc MLP của H1 và nạp trọng số từ checkpoint đã lưu.

    Nhận vào:
        duong_dan (Path): File artifacts/checkpoints/h1_seed<seed>.pt.

    Trả ra:
        Tuple[torch.nn.Module, List[str]]: Mạng ở chế độ eval và danh sách Pa(Y) theo đúng thứ tự đầu vào.
    """
    ck = torch.load(duong_dan, map_location='cpu', weights_only=False)
    meta = ck['meta']
    an = 32
    mang = torch.nn.Module()
    mang.net = torch.nn.Sequential(torch.nn.Linear(meta['in_features'], an), torch.nn.BatchNorm1d(an), torch.nn.ReLU(),
                                   torch.nn.Dropout(0.2), torch.nn.Linear(an, an // 2), torch.nn.ReLU(),
                                   torch.nn.Linear(an // 2, 1))
    mang.load_state_dict(ck['state_dict'])
    mang.eval()
    return mang, list(meta['parents'])


# ==============================================================================================================
# ma_hoa_dau_vao_h1  (def)
# Vị trí: Cell 62 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mã hóa Pa(Y) như H1 (Z1 one-hot 3 chiều, biến nhị phân giữ nguyên) cho phần tính tay ở 8.2c.
# ==============================================================================================================
def ma_hoa_dau_vao_h1(df: pd.DataFrame, cha: List[str]) -> torch.Tensor:
    """
    Hàm này để làm gì: Mã hóa Pa(Y) như H1 (Z1 one-hot 3 chiều, biến nhị phân giữ nguyên).

    Nhận vào:
        df (pd.DataFrame): Các hồ sơ cần truy vấn.
        cha (List[str]): Pa(Y) theo thứ tự đầu vào.

    Trả ra:
        torch.Tensor: Ma trận đặc trưng.
    """
    khoi = [np.eye(3)[df[p].values] if p == 'Z1' else df[[p]].values for p in cha]
    return torch.tensor(np.hstack(khoi).astype(np.float32))


# ==============================================================================================================
# q_b0  (def)
# Vị trí: Cell 62 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính tay q_B0(t, C_i): tra trực tiếp P(Y=1 | T=t, cha khác = giá trị của hồ sơ) trong bảng CPD của
#            B0.
# LÝ THUYẾT: Minh bạch hóa công thức q_m(t, C_i) của Slide 59: đặt T = t cho mọi hồ sơ, giữ nguyên hiệp biến,
#            tra bảng.
# ==============================================================================================================
def q_b0(df: pd.DataFrame, t: int) -> np.ndarray:
    chi_so = tuple([1] + [np.full(len(df), t) if p == 'T' else df[p].values for p in cha_b0])
    return cpd_y.values[chi_so]


# ==============================================================================================================
# q_h1  (def)
# Vị trí: Cell 62 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính tay q_H1(t, C_i): đặt T = t, mã hóa Pa(Y), cho MLP dự đoán và qua sigmoid.
# ==============================================================================================================
def q_h1(df: pd.DataFrame, t: int) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(mlp_h1.net(ma_hoa_dau_vao_h1(df.assign(T=t), cha_h1)).squeeze(-1)).numpy()
