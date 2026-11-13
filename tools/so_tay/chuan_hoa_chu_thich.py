# -*- coding: utf-8 -*-
"""Chuẩn hóa chú thích trong notebook (không chạy lại code, output giữ nguyên):
1. Gỡ toàn bộ chú thích cuối dòng (# đứng sau code) ở các cell mục 8–9.
2. Chèn vài khối chú thích ngắn, đứng riêng dòng: đại ý của hàm và vài biến chính.
3. Kiểm tra AST từng cell không đổi trước khi ghi."""
import ast, io, json, sys, tokenize

CELLS = [45, 47, 49, 51, 52, 57, 75, 77, 79, 83]
BO_DONG = {'# Propensity Score Model', '# Outcome Models', '# Bootstrap 95% Confidence Interval'}

# Mỗi mục: (cell, đoạn neo, chú thích). Đoạn neo là văn bản (có thể nhiều dòng) xuất hiện đúng 1 lần trong cell;
# chú thích được chèn ngay trước dòng đầu của đoạn neo, cùng thụt lề với dòng đó.
CHEN = [
    # ---------- Cell 45: B0 ----------
    (45, '    dag = nx.DiGraph()\n    dag.add_nodes_from(cfg.all_nodes)\n    for z',
     ['# Đại ý: sơ đồ vẽ tay 11 cạnh (5 Z -> T, 5 Z -> Y, T -> Y); chỉ để so với B0, không dùng để fit.']),
    (45, '    cam = [(cfg.outcome, v)',
     ['# Đại ý: cạnh cấm theo thời gian. Y không gây ra gì; T không gây ra Z.']),
    (45, '    model = DiscreteBayesianNetwork(list(dag.edges()))',
     ['# Đại ý: điền bảng CPD cho mọi nút bằng BDeu (ESS = cfg.b0_ess = 5) trên TRAIN.']),
    (45, '    tong = 0.0',
     ['# Đại ý: cộng log P(giá trị | cha) của mọi nút, mọi SV. Càng lớn (ít âm) thì mạng khớp dữ liệu càng tốt.']),
    (45, '    tri_thuc = ExpertKnowledge(',
     ['# Đại ý: thử 3 hàm điểm (BIC, BDeu, K2), mỗi hàm cho một sơ đồ; giữ sơ đồ có log-likelihood VAL cao nhất.',
      '# Biến chính: tri_thuc = cạnh cấm + cạnh bắt buộc T -> Y; ung_vien = bảng 3 ứng viên; tot_nhat = (điểm VAL, tên, DAG).']),
    (45, "    noi_dung = {'nodes'",
     ['# Đại ý: lưu danh sách cạnh ra JSON để nạp lại và kiểm tra sau.']),
    (45, '    cpd_y = model.get_cpds(cfg.outcome)',
     ['# Đại ý: tra bảng CPD của Y cho từng SV, với T ép = 0 và ép = 1.',
      '# Biến chính: q0, q1 = P(Y=1 | do(T=0/1), hồ sơ SV); pred_probs = dùng T thật, để chấm AUROC.']),
    (45, '    idx_0, idx_1, idx_pred = [1], [1], [1]',
     ['# Chỉ số đầu = 1 là cột "Y = 1"; các chỉ số sau là trạng thái của từng nút cha.']),
    (45, 'sample_split = CACHED_SPLITS[42]',
     ['# Chạy thử B0 trên seed 42: học sơ đồ, điền CPD, tính ATE trên TEST.']),
    # ---------- Cell 47: H1 ----------
    (47, 'STATE_CARD: Dict[str, int]',
     ['# Số trạng thái của mỗi biến: Z1 có 3 mức, còn lại nhị phân.']),
    (47, '    khoi: List[np.ndarray] = []',
     ['# Đại ý: Z1 (3 mức) -> one-hot 3 cột; biến nhị phân giữ 1 cột. Với Pa(Y) = {T, Z1, Z2, Z3} ra 6 cột.']),
    (47, '    set_seed(seed)\n    X_train = encode_h1_features',
     ['# Đại ý: huấn luyện MLP dự đoán P(Y=1 | cha của Y). Sau mỗi epoch tính loss trên VAL;',
      '# nếu VAL không giảm suốt cfg.h1_patience = 15 epoch thì dừng sớm và lấy lại trọng số tốt nhất.',
      '# Biến chính: best_val, best_epoch, best_w = loss VAL nhỏ nhất, epoch đó, trọng số đó; train_hist, val_hist = đường loss.']),
    (47, '    duong_dan = Path(cfg.checkpoints_dir)',
     ['# Đại ý: lưu trọng số MLP + nhật ký huấn luyện ra file .pt.']),
    (47, '    b0_y_cpd = b0_model.get_cpds(cfg.outcome)',
     ['# Đại ý: MLP dự đoán P(Y=1) cho mọi tổ hợp cha (16 tổ hợp), ghép thành bảng CPD mới thay cho CPD(Y) của B0.',
      '# Các nút khác giữ nguyên bảng BDeu của B0, nên H1 chỉ khác B0 ở đúng bảng của Y.']),
    (47, '    p_y1 = np.clip(p_y1, 1e-4, 1.0 - 1e-4)',
     ['# Chặn xác suất khỏi 0 và 1 để bảng CPD hợp lệ.']),
    (47, 'sample_parents = [p for p in',
     ['# Chạy thử H1 trên seed 42, dùng đúng sơ đồ B0 vừa học.']),
    # ---------- Cell 49: H2 ----------
    (49, '    nodes = list(cfg.all_nodes)\n    M = np.ones',
     ['# Đại ý: ma trận 7x7, ô [i, j] = 0 nghĩa là cấm cạnh i -> j (cùng danh sách cấm như B0, cộng đường chéo).']),
    (49, "        self.register_buffer('mask'",
     ['# fc1: trọng số tầng 1, kích thước (nút đích, hidden, nút nguồn); luôn nhân với mặt nạ M trước khi dùng.']),
    (49, '    set_seed(seed)\n    nodes = list(cfg.all_nodes)\n    d = len(nodes)',
     ['# Đại ý: tối ưu loss = sai số tái tạo + lambda_l1 * |W| + alpha * h + rho/2 * h^2, với h(W) = 0 khi không có chu trình.',
      '# Vòng ngoài: nếu h giảm chưa đủ thì tăng rho x10, rồi alpha += rho * h; dừng khi h < cfg.h2_h_tol.',
      '# Biến chính: W = độ mạnh cạnh i -> j; h_hist = h qua từng vòng; forbidden_max = trọng số lớn nhất ở ô cấm (phải = 0).']),
    (49, '    nodes = list(cfg.all_nodes)\n    g = nx.DiGraph()',
     ['# Đại ý: giữ cạnh có |W| >= threshold; nếu còn chu trình thì cắt cạnh yếu nhất trong vòng; thiếu T -> Y thì ép thêm (cờ forced).']),
    (49, '    bang: List[Dict[str, Any]] = []',
     ['# Đại ý: thử từng ngưỡng tau, mỗi tau cho một sơ đồ; giữ sơ đồ có log-likelihood VAL cao nhất.']),
    (49, 'lambda_scores: List[Dict[str, Any]] = []',
     ['# Chọn lambda_l1 một lần trên VAL của seed 42, rồi dùng chung cho mọi seed.']),
    (49, 'sample_h2_res = train_notears_mlp(',
     ['# Chạy thử H2 trên seed 42: học W, chọn tau, điền CPD BDeu như B0, tính ATE.']),
    # ---------- Cell 51: hàm đánh giá ----------
    (51, '    best_thresh, best_f1 = 0.5, -1.0',
     ['# Đại ý: ngưỡng phân loại cho Macro-F1 chọn trên VAL (0.2 đến 0.8), rồi áp lên TEST. AUROC, Brier không cần ngưỡng.']),
    (51, '    diffs = q1 - q0',
     ['# Đại ý: lấy mẫu lại các SV của C_test 1,000 lần, mỗi lần tính trung bình (q1 - q0); CI = phân vị 2.5% và 97.5%.']),
    (51, '    try:\n        cm = CausalModel',
     ['# Đại ý: hỏi DoWhy xem trên chính sơ đồ này có tập cửa sau để ước lượng T -> Y không, và tập đó là gì.']),
    (51, '    if not adj_vars:',
     ['# Đại ý: hồ sơ test "được hỗ trợ" nếu trong TRAIN có cả người T=0 và T=1 cùng hồ sơ đó (positivity).']),
    (51, 'prediction_results: List[Dict[str, Any]] = []',
     ['# Ba nơi gom kết quả: Table A (dự báo), Table B (can thiệp), và các mô hình đã học của từng seed.']),
    # ---------- Cell 52: 5 run ghép cặp ----------
    (52, '    t0 = time.time()\n    _, _, prob_val = query_model_effects',
     ['# Đại ý: với một mô hình đã học, tính ATE + CI (Table B) và AUROC, Brier, F1 (Table A), kiểm tra định danh và overlap.',
      '# Biến chính: q0, q1 = xác suất tốt nghiệp khi ép T = 0 / 1 cho từng SV trong C_test; adj = các cha của Y trừ T.']),
    (52, 'for seed in GLOBAL_CONFIG.seeds:',
     ['# Vòng lặp chính: mỗi seed dùng CHUNG một split TRAIN/VAL/TEST cho cả ba mô hình (run ghép cặp).']),
    # ---------- Cell 57: refuter và Table C ----------
    (57, '    cm = CausalModel(data=df[',
     ['# Đại ý: ước lượng ATE bằng hồi quy tuyến tính trên sơ đồ, rồi chạy 2 refuter, mỗi loại 30 lần:',
      '#   placebo: xáo T ngẫu nhiên, ATE phải về gần 0; subset: lấy 80% dữ liệu, ATE phải gần như cũ.']),
    (57, "    parents = list(b0_model.get_cpds('Y')",
     ['# Đại ý: H1 là mạng nơ-ron nên DoWhy không kiểm được; tự làm 2 phép thử tương tự bằng cách huấn luyện lại MLP.',
      '# Biến chính: plc = ATE khi T bị xáo (kỳ vọng ~0); sub = ATE khi chỉ dùng 80% TRAIN (kỳ vọng gần ATE gốc).']),
    (57, 'ref_seed = 42',
     ['# (1) Lấy dữ liệu và sơ đồ B0, H2 của seed 42 (seed đầu tiên).']),
    (57, "print('Đang chạy refuter DoWhy",
     ['# (2) Chạy refuter cho ba mô hình.']),
    (57, 'refuter_map = {',
     ['# (3) Gom kết quả: Original = ATE gốc, Placebo = ATE khi xáo T, Subset = ATE trên 80% dữ liệu.']),
    (57, 'table_c_rows = []',
     ['# (4) Mỗi mô hình một dòng Table C. Đạt nếu |Placebo| < 0.05 và |Subset - Original| < 0.05;',
      '#     kèm ATE trung bình ± độ lệch qua 5 seed, độ tản max - min, thời gian chạy.']),
    # ---------- Cell 75: AIPW ----------
    (75, 'ps_model = LogisticRegression(',
     ['# (1) Xác suất được học bổng theo 7 hiệp biến (propensity score), chặn trong [0.05, 0.95].']),
    (75, 'm0 = LogisticRegression(',
     ['# (2) Hai mô hình kết cục "ở lại trường": m0 học trên nhóm không học bổng, m1 trên nhóm có học bổng.',
      '#     mu0, mu1 = dự đoán ở lại cho MỌI SV nếu không / có học bổng.']),
    (75, "t_full = df_full['T']",
     ['# (3) Điểm AIPW = dự đoán + phần sửa sai có trọng số 1/ps. Vẫn đúng nếu một trong hai mô hình đúng.']),
    (75, 'np.random.seed(ADV_CONFIG.seed)',
     ['# (4) Bootstrap 1,000 lần trên 4,424 SV để có CI 95%.']),
    # ---------- Cell 77: trung gian ----------
    (77, 'df_med = df_completed.copy()',
     ['# M = số tín chỉ đạt HK1 (trung gian), Y = tốt nghiệp. Dùng 3,630 SV đã có kết cục.']),
    (77, "nde = model_y_lpm.params['T']",
     ['# NDE = hệ số T khi đã giữ M cố định (tác dụng trực tiếp); NIE = (T -> M) x (M -> Y) (tác dụng qua HK1).']),
    # ---------- Cell 79: CATE ----------
    (79, 'subgroups_specs = [',
     ['# 6 nhóm định trước (tuổi, nợ học phí, nơi ở). Mỗi nhóm: (tên, điều kiện lọc).']),
    (79, 'for label, cond_fn in subgroups_specs:',
     ['# Mỗi nhóm: khớp logistic Y ~ T + hiệp biến, rồi ép T = 0 và T = 1 cho mọi người trong nhóm (G-computation).',
      '# p0, p1 = tỷ lệ tốt nghiệp dự đoán khi không / có học bổng; CATE = p1 - p0.']),
    # ---------- Cell 83: RV ----------
    (83, '@dataclass',
     ['# Gói kết quả: hệ số và t của học bổng, bậc tự do, partial R2 của T với Y, RV và RV ở mức 5%.']),
    (83, '    """\n    Z = pd.DataFrame({\n        \'Z1\': pd.qcut(du_lieu[\'Age_at_enrollment\'], q=3, labels=[0, 1, 2]).astype(int),\n'
         '        \'Z2\': (du_lieu[\'Admission_grade\'] >= du_lieu[\'Admission_grade\'].median()).astype(int),\n'
         '        \'Z3\': (du_lieu[\'Mothers_qualification\'] > 1).astype(int),\n'
         '        \'Z4\': du_lieu[\'Debtor\'].astype(int),\n'
         '        \'Z5\': du_lieu[\'Tuition_fees_up_to_date\'].astype(int),\n    })\n    X = np.column_stack',
     None),  # neo có dòng docstring, xử lý bằng lệch 1 dòng bên dưới
    (83, '    f = abs(t_stat) / np.sqrt(bac_tu_do)',
     ['# Công thức Cinelli–Hazlett: f = |t| / sqrt(df); RV = 0.5 * (sqrt(f^4 + 4 f^2) - f^2).']),
    (83, '    mo_ta = {',
     ['# Đại ý: bỏ lần lượt từng biến nền, xem sai số tăng bao nhiêu -> biến đó "mạnh" cỡ nào. Dùng làm mốc so với RV.']),
    (83, '    day_du = rss(list(Z.columns))',
     ['# Partial R2 của biến c = (RSS khi bỏ c - RSS đầy đủ) / RSS khi bỏ c: phần biến c giải thích thêm cho Y.']),
]
# Neo đặc biệt: chú thích chèn sau dòng đầu của đoạn neo (bỏ qua dòng đóng docstring)
CHEN_LECH = {(83, 4): ['# Đại ý: hồi quy tuyến tính Y ~ T + 5 biến nền (mã hóa riêng cho mục này), lấy t của học bổng, rồi tính RV.']}


def strip_trailing(src):
    lines = src.split('\n')
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT and not t.line.strip().startswith('#'):
            r, c = t.start
            lines[r - 1] = lines[r - 1][:c].rstrip()
    return '\n'.join(l for l in lines if l.strip() not in BO_DONG)


def chen(src, cell):
    lines = src.split('\n')
    tac_vu = []
    for stt, (c, neo, cm) in enumerate(CHEN):
        if c != cell:
            continue
        assert src.count(neo) == 1 and (src.startswith(neo) or ('\n' + neo) in src), (cell, neo[:70], src.count(neo))
        k = src[:src.index(neo)].count('\n')
        if cm is None:  # neo bắt đầu bằng dòng đóng docstring: chèn trước dòng kế tiếp
            cm, k = CHEN_LECH[(cell, 4)], k + 1
        tac_vu.append((k, cm))
    for k, cm in sorted(tac_vu, key=lambda x: -x[0]):
        thut = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        lines[k:k] = [thut + x for x in cm]
    return '\n'.join(lines), len(tac_vu)


for path in sys.argv[1:]:
    nb = json.load(open(path, encoding='utf-8'))
    bao_cao = []
    for i in CELLS:
        c = nb['cells'][i]
        goc = ''.join(c['source'])
        moi, n = chen(strip_trailing(goc), i)
        assert ast.dump(ast.parse(moi)) == ast.dump(ast.parse(goc)), f'cell {i}: AST thay đổi!'
        c['source'] = moi.splitlines(keepends=True)
        bao_cao.append((i, n))
    json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(path, 'a', encoding='utf-8').write('\n')
    print(path, '-> (cell, số khối chú thích):', bao_cao)
