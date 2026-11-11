# -*- coding: utf-8 -*-
"""
NHÓM 09 — Đánh giá dùng chung, vòng lặp 5 run ghép cặp, xuất Table A và B (mục 8.E)

Chỉ số dự báo (ngưỡng chọn trên VAL), CI bootstrap của ATE, định danh DoWhy trên DAG của từng mô hình, kiểm
tra overlap, vòng lặp 5 seed và tổng hợp Table A, Table B.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - evaluate_prediction_metrics                Cell 51 (def)
  - compute_bootstrap_ate_ci                   Cell 51 (def)
  - identify_with_dowhy                        Cell 51 (def)
  - overlap_report                             Cell 51 (def)
  - record_model_run                           Cell 52 (def)
  - doan_ma_cell_52_vong_lap_5_run_ghep_cap    Cell 52 (đoạn mã)
  - mean_std                                   Cell 55 (def)
  - doan_ma_cell_55_xuat_table_a_b             Cell 55 (đoạn mã)
"""


# ==============================================================================================================
# evaluate_prediction_metrics  (def)
# Vị trí: Cell 51 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính chỉ số dự báo trên TEST (Slide 74): AUROC, Macro-F1, Brier, log-loss, precision, recall.
# CÁCH LÀM :
#   1. Dò 31 ngưỡng từ 0.2 đến 0.8 trên VALIDATION, chọn ngưỡng có Macro-F1 cao nhất.
#   2. Áp ngưỡng đó cho TEST để tính Macro-F1, precision, recall.
#   3. AUROC, Brier, log-loss tính trực tiếp trên xác suất (không cần ngưỡng).
# ❓ HỎI   : Vì sao Macro-F1 của B0 và H1 trùng nhau?
#   ➜ ĐÁP : Pa(Y) chỉ có 16 cấu hình, hai mô hình xếp hạng giống nhau tại ngưỡng chọn, nên gán nhãn giống hệt;
#           khác biệt chỉ ở xác suất (AUROC, Brier).
# ==============================================================================================================
def evaluate_prediction_metrics(y_true: np.ndarray, y_prob: np.ndarray, val_y_true: np.ndarray,
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


# ==============================================================================================================
# compute_bootstrap_ate_ci  (def)
# Vị trí: Cell 51 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : CI 95% của ATE bằng percentile bootstrap: lấy mẫu lại có hoàn lại các hồ sơ C_test, GIỮ CỐ ĐỊNH mô
#            hình.
# CÁCH LÀM :
#   1. Tính hiệu từng hồ sơ d_i = q1_i - q0_i.
#   2. Lặp 1000 lần: lấy mẫu lại d có hoàn lại, tính trung bình.
#   3. Lấy phân vị 2.5% và 97.5%.
# LƯU Ý    : CI này chỉ phản ánh bất định do chọn quần thể tham chiếu; chưa gồm bất định tham số hay đồ thị. Độ
#            dao động giữa 5 seed bổ sung phần đó.
# ❓ HỎI   : CI hẹp (khoảng ±0.008) có nghĩa ATE rất chính xác?
#   ➜ ĐÁP : Không hẳn: mô hình được giữ cố định. Bất định do học lại mô hình thể hiện qua độ tản giữa 5 seed
#           (Table C).
# ==============================================================================================================
def compute_bootstrap_ate_ci(q0: np.ndarray, q1: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """
    95% CI percentile bootstrap: lấy mẫu lại các hồ sơ tham chiếu C_test (có hoàn lại), GIỮ CỐ ĐỊNH mô hình đã khớp.
    Đây là bất định do lấy mẫu quần thể tham chiếu, chưa gồm bất định tham số/đồ thị (Slide 68).
    """
    diffs = q1 - q0  # hiệu "có - không học bổng" của từng hồ sơ TEST
    boot = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]  # 1000 lần: bốc lại 545 hồ sơ có hoàn lại, tính ATE
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))  # phân vị 2.5% và 97.5% (alpha = 0.05)


# ==============================================================================================================
# identify_with_dowhy  (def)
# Vị trí: Cell 51 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Chạy DoWhy identify_effect trên CHÍNH DAG của từng mô hình (Slide 57, 62) để xác nhận hiệu ứng định
#            danh được.
# ĐẦU RA   : Dict: status ("identified (backdoor)" hoặc lỗi), backdoor_set, text.
# LƯU Ý    : proceed_when_unidentifiable=False: nếu không định danh được thì báo lỗi thay vì đoán.
# ==============================================================================================================
def identify_with_dowhy(dag: nx.DiGraph, train_df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Chạy DoWhy identify_effect trên CHÍNH DAG của mô hình (Slide 57, 62); trả về trạng thái và tập cửa sau."""
    try:
        cm = CausalModel(data=train_df[list(cfg.all_nodes)], treatment=cfg.treatment, outcome=cfg.outcome, graph=dag)
        est = cm.identify_effect(proceed_when_unidentifiable=False)
        bd = est.get_backdoor_variables()
        return {'status': 'identified (backdoor)', 'backdoor_set': sorted(bd), 'text': str(est)}
    except Exception as exc:
        return {'status': f'NOT identified: {type(exc).__name__}', 'backdoor_set': [], 'text': str(exc)}


# ==============================================================================================================
# overlap_report  (def)
# Vị trí: Cell 51 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Đếm hồ sơ C_test "không được hỗ trợ": trong TRAIN không có ai cùng cấu hình biến điều chỉnh ở CẢ
#            T=0 và T=1.
# CÁCH LÀM :
#   1. groupby (biến điều chỉnh, T) trên TRAIN, đếm số người.
#   2. Cấu hình được hỗ trợ nếu đếm ở T=0 và T=1 đều > 0.
#   3. Đếm hồ sơ test có cấu hình không được hỗ trợ.
# ❓ HỎI   : Hồ sơ không được hỗ trợ thì xử lý sao?
#   ➜ ĐÁP : Được ghi rõ ở cột Overlap_Status (1–2% ở H2); ước lượng ở các hồ sơ đó dựa vào làm mịn BDeu.
# ==============================================================================================================
def overlap_report(train_df: pd.DataFrame, test_df: pd.DataFrame, adj_vars: List[str], cfg: Config) -> Dict[str, Any]:
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


# ==============================================================================================================
# record_model_run  (def)
# Vị trí: Cell 52 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Đánh giá một mô hình trong một seed: dự báo trên TEST, can thiệp chuẩn hóa trên C_test, CI, định
#            danh, overlap; ghi vào hai danh sách kết quả.
# CÁCH LÀM :
#   1. query_model_effects trên VAL (để chọn ngưỡng F1) và trên TEST (q0, q1, xác suất dự báo).
#   2. ATE = trung bình q1 - trung bình q0; CI bootstrap.
#   3. evaluate_prediction_metrics: chỉ số dự báo, dùng Y_test.
#   4. Tập điều chỉnh = Pa(Y) bỏ T; identify_with_dowhy; overlap_report.
#   5. Thêm một dòng vào prediction_results (Table A) và một dòng vào intervention_results (Table B).
# LÝ THUYẾT: Y_test CHỈ dùng để chấm dự báo (Table A). ATE (Table B) chỉ dùng hiệp biến C_test, không dùng
#            Y_test (Slide 56).
# ==============================================================================================================
def record_model_run(model_name: str, seed: int, graph_id: str, dag: nx.DiGraph, bn: DiscreteBayesianNetwork,
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


# ==============================================================================================================
# doan_ma_cell_52_vong_lap_5_run_ghep_cap  (đoạn mã tuần tự)
# Vị trí: Cell 52 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Vòng lặp chính: với mỗi seed 42..46, khớp B0, H1, H2 trên cùng split rồi đánh giá bằng
#            record_model_run.
# CÁCH LÀM :
#   1. B0: learn_b0_dag + fit_bn_on_dag, đo thời gian, đánh giá, lưu JSON đồ thị.
#   2. H1: lấy Pa(Y) của B0, train_h1_mlp, build_h1_bayesian_network, lưu checkpoint, đánh giá.
#   3. H2: train_notears_mlp với lambda đã chọn, select_h2_graph, fit_bn_on_dag, đánh giá, lưu W và JSON (kể cả
#      log h(W) từng vòng).
#   4. assert không có trọng số khác 0 ở ô bị cấm.
#   5. Giữ mô hình, dữ liệu và nhật ký trong run_artifacts[seed] cho mục 8.E2.
# LÝ THUYẾT: Ghép cặp: trong cùng một seed, ba mô hình dùng cùng TRAIN/VAL/TEST, nên hiệu B0 - H1 - H2 là so
#            sánh theo cặp (Slide 68).
# ==============================================================================================================
def doan_ma_cell_52_vong_lap_5_run_ghep_cap():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    for seed in GLOBAL_CONFIG.seeds:
        print(f'\n--- Paired Run, seed {seed} ---')
        set_seed(seed)
        split = CACHED_SPLITS[seed]
        train_data = PRIMARY_DF.iloc[split['train_indices']].reset_index(drop=True)
        val_data = PRIMARY_DF.iloc[split['val_indices']].reset_index(drop=True)
        test_data = PRIMARY_DF.iloc[split['test_indices']].reset_index(drop=True)
        art: Dict[str, Any] = {}

        # 1. B0: học DAG bằng HillClimb trên TRAIN, chọn hàm điểm trên VALIDATION, khớp CPD BDeu
        t0 = time.time()
        b0_dag, b0_score, b0_cands = learn_b0_dag(train_data, val_data, GLOBAL_CONFIG)
        b0_model = fit_bn_on_dag(train_data, b0_dag, GLOBAL_CONFIG)
        b0_fit = time.time() - t0
        b0_id = f'B0-HC-{b0_score}-s{seed} ({len(b0_dag.edges())} edges)'
        art['B0'] = record_model_run('B0', seed, b0_id, b0_dag, b0_model, train_data, val_data, test_data, b0_fit, GLOBAL_CONFIG)
        save_graph_artifacts(b0_dag, f'b0_dag_seed{seed}', GLOBAL_CONFIG,
                             {'score_selected_on_val': b0_score, 'candidates': b0_cands,
                              'identification': art['B0']['identification']})

        # 2. H1: cùng DAG B0, thay CPD của Y bằng MLP (đầu vào = Pa(Y))
        t0 = time.time()
        parents_y = list(b0_model.get_cpds('Y').variables[1:])
        h1_mlp, h1_log = train_h1_mlp(train_data, val_data, parents_y, GLOBAL_CONFIG, seed=seed)
        h1_model = build_h1_bayesian_network(b0_model, h1_mlp, GLOBAL_CONFIG)
        h1_fit = time.time() - t0
        save_h1_checkpoint(h1_mlp, h1_log, seed, GLOBAL_CONFIG)
        art['H1'] = record_model_run('H1', seed, b0_id + ' + neural CPD(Y)', b0_dag, h1_model, train_data, val_data,
                                     test_data, h1_fit, GLOBAL_CONFIG)
        art['H1_log'] = h1_log
        art['H1_mlp'] = h1_mlp

        # 3. H2: NOTEARS-MLP + mặt nạ M, ngưỡng chọn trên VALIDATION, CPD BDeu như B0
        t0 = time.time()
        h2_res = train_notears_mlp(train_data, GLOBAL_CONFIG, seed=seed, lambda_l1=H2_LAMBDA)
        h2_dag, h2_tau, h2_tab = select_h2_graph(h2_res, train_data, val_data, GLOBAL_CONFIG)
        h2_model = fit_bn_on_dag(train_data, h2_dag, GLOBAL_CONFIG)
        h2_fit = time.time() - t0
        h2_id = f'H2-NOTEARS-MLP-s{seed} (tau={h2_tau}, {len(h2_dag.edges())} edges, h={h2_res["h_final"]:.1e})'
        art['H2'] = record_model_run('H2', seed, h2_id, h2_dag, h2_model, train_data, val_data, test_data, h2_fit, GLOBAL_CONFIG)
        np.save(Path(GLOBAL_CONFIG.graphs_dir) / f'h2_weights_seed{seed}.npy', h2_res['W'])
        raw_dag, raw_cut, _ = graph_from_weights(h2_res['W'], h2_tau, GLOBAL_CONFIG)
        save_graph_artifacts(h2_dag, f'h2_dag_seed{seed}', GLOBAL_CONFIG,
                             {'tau_selected_on_val': h2_tau, 'lambda_l1': H2_LAMBDA, 'h_final': h2_res['h_final'],
                              'converged': h2_res['converged'], 'outer_iters': h2_res['outer_iters'],
                              'h_history': h2_res['h_history'], 'recon_history': h2_res['recon_history'],  # log từng vòng ngoài
                              'forbidden_weight_max': h2_res['forbidden_weight_max'], 'edges_cut_for_cycles': raw_cut,
                              'tau_table': h2_tab, 'identification': art['H2']['identification']})
        assert h2_res['forbidden_weight_max'] == 0.0, 'Vi phạm mặt nạ: có trọng số khác 0 ở cạnh bị cấm!'
        art['H2_res'] = h2_res
        art['models'] = {'B0': b0_model, 'H1': h1_model, 'H2': h2_model}  # giữ lại để chẩn đoán overfit ở mục 8.E2
        art['data'] = {'TRAIN': train_data, 'VAL': val_data, 'TEST': test_data}
        run_artifacts[seed] = art

        ates = {m: np.mean(art[m]['q1']) - np.mean(art[m]['q0']) for m in ('B0', 'H1', 'H2')}
        print(f"B0[{b0_score}, {len(b0_dag.edges())} cạnh] ATE={ates['B0']:+.4f} | H1 ATE={ates['H1']:+.4f} | "
              f"H2[tau={h2_tau}, hội tụ={h2_res['converged']}] ATE={ates['H2']:+.4f}")
    print('\n=== TOÀN BỘ 5 PAIRED RUNS ĐÃ HOÀN TẤT ===')


# ==============================================================================================================
# mean_std  (def)
# Vị trí: Cell 55 · mục 8.E2 Quá trình huấn luyện và kiểm tra overfit (đọc Hình 8.2 và Bảng 8.E2)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Định dạng chuỗi "trung bình ± độ lệch chuẩn" cho dòng tổng hợp 5 seed của Tables A–C.
# ==============================================================================================================
def mean_std(s: pd.Series, fmt: str = '.4f') -> str:
    return f"{s.mean():{fmt}} ± {s.std():{fmt.replace('+', '')}}"


# ==============================================================================================================
# doan_ma_cell_55_xuat_table_a_b  (đoạn mã tuần tự)
# Vị trí: Cell 55 · mục 8.E2 Quá trình huấn luyện và kiểm tra overfit (đọc Hình 8.2 và Bảng 8.E2)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Xuất Table A (dự báo) và Table B (can thiệp): tính hiệu theo cặp so với B0 trong cùng seed, thêm
#            dòng tổng hợp trung bình ± độ lệch chuẩn, ghi CSV.
# ==============================================================================================================
def doan_ma_cell_55_xuat_table_a_b():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    df_pred_raw = pd.DataFrame(prediction_results)
    df_interv_raw = pd.DataFrame(intervention_results)
    for metric in ['AUROC', 'Macro_F1', 'Brier_Score']:
        b0_map = df_pred_raw[df_pred_raw['Model'] == 'B0'].set_index('Seed')[metric]
        df_pred_raw[f'Delta_{metric}_vs_B0'] = df_pred_raw[metric] - df_pred_raw['Seed'].map(b0_map)
    b0_ate = df_interv_raw[df_interv_raw['Model'] == 'B0'].set_index('Seed')['ATE']
    df_interv_raw['Delta_ATE_vs_B0'] = df_interv_raw['ATE'] - df_interv_raw['Seed'].map(b0_ate)
    summary_rows_a = []
    for m in ['B0', 'H1', 'H2']:
        sub = df_pred_raw[df_pred_raw['Model'] == m]
        summary_rows_a.append({
            'Model': f'{m} (Mean ± Std)', 'Seed': 'Pooled 5 Runs', 'Graph': 'per-seed (see rows)',
            'Test_Size': int(sub['Test_Size'].iloc[0]), 'Test_Positive_Rate': f"{sub['Test_Positive_Rate'].mean():.4f}",
            **{c: mean_std(sub[c]) for c in ['AUROC', 'Macro_F1', 'Brier_Score', 'Log_Loss', 'Precision', 'Recall']},
            'F1_Threshold_from_Val': mean_std(sub['F1_Threshold_from_Val'], '.2f'),
            **{f'Delta_{c}_vs_B0': (mean_std(sub[f'Delta_{c}_vs_B0'], '+.4f') if m != 'B0' else '---')
               for c in ['AUROC', 'Macro_F1', 'Brier_Score']},
            'Fit_Time_s': f"{sub['Fit_Time_s'].mean():.2f}s", 'Query_Time_s': f"{sub['Query_Time_s'].mean():.4f}s",
        })
    df_table_a = pd.concat([df_pred_raw, pd.DataFrame(summary_rows_a)], ignore_index=True)
    df_table_a['Hardware'] = HARDWARE
    df_table_a.to_csv(Path(GLOBAL_CONFIG.tables_dir) / 'Table_A_Prediction.csv', index=False)
    summary_rows_b = []
    for m in ['B0', 'H1', 'H2']:
        sub = df_interv_raw[df_interv_raw['Model'] == m]
        summary_rows_b.append({
            'Model': f'{m} (Mean ± Std)', 'Seed': 'Pooled 5 Runs', 'Reference_Cohort': 'C_test per seed',
            'Cohort_Size': int(sub['Cohort_Size'].iloc[0]), 'Contrast': 'do(T=1) vs do(T=0)',
            'Risk_do_T0': mean_std(sub['Risk_do_T0']), 'Risk_do_T1': mean_std(sub['Risk_do_T1']),
            'ATE': mean_std(sub['ATE'], '+.4f'),
            'CI_95_Low': f"{sub['CI_95_Low'].mean():+.4f}", 'CI_95_High': f"{sub['CI_95_High'].mean():+.4f}",
            'CI_Method': CI_METHOD + ' (mean of per-seed bounds)', 'Formula': sub['Formula'].iloc[0],
            'Adjustment_Set': ' / '.join(sorted(sub['Adjustment_Set'].unique())),
            'DoWhy_Identification': ' / '.join(sorted(sub['DoWhy_Identification'].unique())),
            'DoWhy_Backdoor_Set': ' / '.join(sorted(sub['DoWhy_Backdoor_Set'].unique())),
            'Overlap_Status': ' / '.join(sorted(sub['Overlap_Status'].unique())),
            'Unsupported_Test_Profiles_pct': f"{sub['Unsupported_Test_Profiles_pct'].mean():.2f}",
            'Delta_ATE_vs_B0': (mean_std(sub['Delta_ATE_vs_B0'], '+.4f') if m != 'B0' else '---'),
        })
    df_table_b = pd.concat([df_interv_raw, pd.DataFrame(summary_rows_b)], ignore_index=True)
    df_table_b.to_csv(Path(GLOBAL_CONFIG.tables_dir) / 'Table_B_Intervention.csv', index=False)
    print('[Xuất bản] Table A, Table B đã lưu trong', GLOBAL_CONFIG.tables_dir)
