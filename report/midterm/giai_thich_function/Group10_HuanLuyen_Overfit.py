# -*- coding: utf-8 -*-
"""
NHÓM 10 — Quá trình huấn luyện, kiểm tra overfit, có / không early stopping (mục 8.E2)

Vẽ đường cong loss của H1 và hội tụ của H2 (Hình 8.2), bảng AUROC/log-loss trên TRAIN/VAL/TEST (Bảng 8.E2),
huấn luyện H1 không dừng sớm để so sánh và lưu 2 checkpoint (Hình 8.2b, Bảng 8.E2b).

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - ve_qua_trinh_huan_luyen                    Cell 53 (def)
  - bang_overfit_theo_tap                      Cell 53 (def)
  - so_sanh_early_stopping                     Cell 53 (def)
  - ve_early_stopping                          Cell 53 (def)
"""


# ==============================================================================================================
# ve_qua_trinh_huan_luyen  (def)
# Vị trí: Cell 53 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 8.2: (a) loss TRAIN/VAL của H1 theo epoch, chấm tại epoch tốt nhất; (b) h(W) của H2 theo vòng
#            ngoài (thang log); (c) sai số tái tạo của H2.
# LƯU Ý    : B0 không có đường cong vì không học bằng gradient.
# ==============================================================================================================
def ve_qua_trinh_huan_luyen(run_artifacts: Dict[int, Dict[str, Any]], cfg: Config, duong_dan_luu: Path) -> None:
    """
    Hàm này để làm gì: Vẽ quá trình huấn luyện của hai mô hình có tối ưu lặp:
    (a) H1 — BCE trên TRAIN và VALIDATION theo từng epoch, đánh dấu epoch tốt nhất (early stopping);
    (b) H2 — ràng buộc phi chu trình h(W) theo từng vòng ngoài của augmented Lagrangian (thang log);
    (c) H2 — sai số tái tạo (MSE/2) trên TRAIN theo từng vòng ngoài.
    B0 không có epoch (CPD BDeu là đếm tần suất có làm mịn) nên không có đường cong.

    Nhận vào:
        run_artifacts (Dict[int, Dict[str, Any]]): Kết quả 5 run ghép cặp của mục 8.E.
        cfg (Config): Cấu hình chung (patience, ngưỡng hội tụ h).
        duong_dan_luu (Path): Đường dẫn lưu PNG.
    """
    mau = dict(zip(cfg.seeds, ['#2b6cb0', '#c05621', '#2f855a', '#6b46c1', '#b83280']))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))
    for seed, art in run_artifacts.items():
        log = art['H1_log']
        ep = np.arange(1, len(log['train_loss']) + 1)
        ax1.plot(ep, log['train_loss'], color=mau[seed], lw=1.4, label=f'seed {seed} TRAIN')
        ax1.plot(ep, log['val_loss'], color=mau[seed], lw=1.4, ls='--', label=f'seed {seed} VAL')
        ax1.scatter([log['best_epoch'] + 1], [log['best_val_bce']], color=mau[seed], s=45, zorder=5, edgecolor='black')
        res = art['H2_res']
        vong = np.arange(1, len(res['h_history']) + 1)
        ax2.semilogy(vong, res['h_history'], marker='o', ms=4, color=mau[seed], label=f'seed {seed}')
        ax3.plot(vong, res['recon_history'], marker='o', ms=4, color=mau[seed], label=f'seed {seed}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE loss')
    ax1.set_title(f'(a) H1: loss TRAIN (liền) / VAL (đứt)\nchấm = epoch tốt nhất, dừng sau {cfg.h1_patience} epoch không cải thiện')
    ax1.legend(fontsize=7, ncol=2)
    ax2.axhline(cfg.h2_h_tol, color='#c53030', ls=':', lw=1.2)
    ax2.text(1, cfg.h2_h_tol * 1.6, f'ngưỡng hội tụ h < {cfg.h2_h_tol:g}', color='#c53030', fontsize=8)
    ax2.set_xlabel('Vòng ngoài (augmented Lagrangian)')
    ax2.set_ylabel('h(W) (thang log)')
    ax2.set_title('(b) H2: ràng buộc phi chu trình h(W) giảm về ~0')
    ax2.legend(fontsize=8)
    ax3.set_xlabel('Vòng ngoài (augmented Lagrangian)')
    ax3.set_ylabel('Sai số tái tạo trên TRAIN')
    ax3.set_title('(c) H2: sai số tái tạo theo vòng ngoài')
    ax3.legend(fontsize=8)
    fig.suptitle('Hình 8.2. Quá trình huấn luyện của H1 và H2 trên 5 seed', fontsize=11)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================================================
# bang_overfit_theo_tap  (def)
# Vị trí: Cell 53 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng 8.E2: AUROC, accuracy (ngưỡng 0.5) và log-loss của B0/H1/H2 trên TRAIN, VAL, TEST cho cả 5
#            seed.
# CÁCH LÀM :
#   1. Với mỗi seed, mô hình và tập: lấy xác suất dự báo bằng query_model_effects, tính 3 chỉ số.
#   2. Tóm tắt trung bình ± độ lệch chuẩn và chênh AUROC TRAIN - TEST.
# LƯU Ý    : Chỉ ĐỌC mô hình đã huấn luyện, không chọn hay chỉnh gì theo TEST; AUROC TEST trùng Table A.
# ❓ HỎI   : Có overfit không?
#   ➜ ĐÁP : Chênh AUROC TRAIN - TEST: B0 +0.002, H1 khoảng 0, H2 +0.008 (nhỏ hơn độ dao động giữa seed ±0.016)
#           => không overfit đáng kể.
# ==============================================================================================================
def bang_overfit_theo_tap(run_artifacts: Dict[int, Dict[str, Any]], cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hàm này để làm gì: Tính AUROC, Accuracy (ngưỡng 0.5) và log-loss của B0, H1, H2 trên cả TRAIN, VALIDATION
    và TEST của từng seed, dùng đúng hàm dự báo P(Y=1 | T_obs, Pa(Y)) của Table A. Chênh lệch TRAIN − TEST lớn
    là dấu hiệu overfit. Hàm này chỉ ĐỌC mô hình đã huấn luyện, không chọn hay chỉnh gì theo TEST.

    Nhận vào:
        run_artifacts (Dict[int, Dict[str, Any]]): Kết quả 5 run ghép cặp (có mô hình và dữ liệu từng tập).
        cfg (Config): Cấu hình chung.

    Trả ra:
        Tuple[pd.DataFrame, pd.DataFrame]: (bảng chi tiết từng seed, bảng tóm tắt trung bình ± độ lệch chuẩn).
    """
    dong: List[Dict[str, Any]] = []
    for seed, art in run_artifacts.items():
        for ten_mo_hinh, mo_hinh in art['models'].items():
            for ten_tap, df in art['data'].items():
                _, _, p = query_model_effects(mo_hinh, df, cfg)
                y = df[cfg.outcome].values.astype(int)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                dong.append({'Model': ten_mo_hinh, 'Seed': seed, 'Tập': ten_tap,
                             'AUROC': metrics.roc_auc_score(y, p),
                             'Accuracy': metrics.accuracy_score(y, (p >= 0.5).astype(int)),
                             'LogLoss': metrics.log_loss(y, p)})
    chi_tiet = pd.DataFrame(dong)
    tom_tat = []
    for ten_mo_hinh in ['B0', 'H1', 'H2']:
        d = chi_tiet[chi_tiet['Model'] == ten_mo_hinh]
        hang: Dict[str, Any] = {'Model': ten_mo_hinh}
        for chi_so in ['AUROC', 'Accuracy', 'LogLoss']:
            for ten_tap in ['TRAIN', 'VAL', 'TEST']:
                x = d[d['Tập'] == ten_tap][chi_so]
                hang[f'{chi_so}_{ten_tap}'] = f'{x.mean():.4f} ± {x.std():.4f}'
        gap = (d[d['Tập'] == 'TRAIN'].set_index('Seed')['AUROC'] - d[d['Tập'] == 'TEST'].set_index('Seed')['AUROC'])
        hang['AUROC_gap_TRAIN−TEST'] = f'{gap.mean():+.4f}'
        tom_tat.append(hang)
    return chi_tiet, pd.DataFrame(tom_tat)


# ==============================================================================================================
# so_sanh_early_stopping  (def)
# Vị trí: Cell 53 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Huấn luyện thêm bản H1 KHÔNG early stopping (đủ 150 epoch) cho mỗi seed, lưu 2 checkpoint riêng và
#            so sánh với bản có early stopping.
# CÁCH LÀM :
#   1. Lưu trạng thái bộ sinh số ngẫu nhiên (random, numpy, torch).
#   2. Lưu bản có early stopping (chính là H1 của Tables A–C) thành h1_seed<s>_early_stop.pt.
#   3. Huấn luyện bản dung_som=False, lưu h1_seed<s>_no_early_stop.pt, dựng mạng Bayes H1 tương ứng.
#   4. Tính AUROC, log-loss trên TRAIN/VAL/TEST và ATE trên TEST cho cả hai bản.
#   5. Khôi phục trạng thái ngẫu nhiên để các cell sau (refuter, Table C) không đổi.
# ❓ HỎI   : Bỏ early stopping thì sao?
#   ➜ ĐÁP : Loss VAL tăng (0.504 so với 0.500), AUROC TEST giảm nhẹ (0.799 so với 0.801): dấu hiệu bắt đầu
#           overfit. ATE trung bình gần như không đổi nhưng từng seed lệch tới ±0.02.
# ==============================================================================================================
def so_sanh_early_stopping(run_artifacts: Dict[int, Dict[str, Any]], cfg: Config) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Hàm này để làm gì: Với mỗi seed, đặt cạnh nhau hai bản H1 trên cùng DAG B0, cùng TRAIN/VAL và cùng seed:
    (1) có early stopping — chính là H1 dùng trong Tables A–C; (2) không early stopping — chạy đủ cfg.h1_epochs
    và giữ trọng số epoch cuối. Lưu hai checkpoint riêng: h1_seed<s>_early_stop.pt và h1_seed<s>_no_early_stop.pt.
    Trạng thái bộ sinh số ngẫu nhiên được lưu và khôi phục, nên các cell phía sau (refuter, Table C) không đổi.

    Nhận vào:
        run_artifacts (Dict[int, Dict[str, Any]]): Kết quả 5 run ghép cặp của mục 8.E.
        cfg (Config): Cấu hình chung.

    Trả ra:
        Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]: (bảng so sánh từng seed, nhật ký huấn luyện bản không dừng sớm).
    """
    trang_thai = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    dong: List[Dict[str, Any]] = []
    nhat_ky_khong_dung: Dict[int, Dict[str, Any]] = {}
    for seed, art in run_artifacts.items():
        d = art['data']
        b0 = art['models']['B0']
        parents = list(b0.get_cpds(cfg.outcome).variables[1:])
        p_es = save_h1_checkpoint(art['H1_mlp'], art['H1_log'], seed, cfg, hau_to='_early_stop')
        mlp_no, log_no = train_h1_mlp(d['TRAIN'], d['VAL'], parents, cfg, seed=seed, dung_som=False)
        p_no = save_h1_checkpoint(mlp_no, log_no, seed, cfg, hau_to='_no_early_stop')
        nhat_ky_khong_dung[seed] = log_no
        ban = [('Có early stopping', art['models']['H1'], art['H1_log'], p_es),
               ('Không early stopping', build_h1_bayesian_network(b0, mlp_no, cfg), log_no, p_no)]
        for ten, bn, log, duong_dan in ban:
            q0, q1, _ = query_model_effects(bn, d['TEST'], cfg)
            hang: Dict[str, Any] = {'Seed': seed, 'Bản H1': ten, 'Epoch_đã_chạy': log['epochs_run'],
                                    'Trọng_số_lấy_ở_epoch': (log['best_epoch'] + 1) if log['early_stopping'] else log['epochs_run']}
            for ten_tap in ['TRAIN', 'VAL', 'TEST']:
                _, _, p = query_model_effects(bn, d[ten_tap], cfg)
                y = d[ten_tap][cfg.outcome].values.astype(int)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                hang[f'AUROC_{ten_tap}'] = metrics.roc_auc_score(y, p)
                hang[f'LogLoss_{ten_tap}'] = metrics.log_loss(y, p)
            hang['ATE_TEST'] = float(np.mean(q1) - np.mean(q0))
            hang['Checkpoint'] = Path(duong_dan).name
            dong.append(hang)
    random.setstate(trang_thai[0])
    np.random.set_state(trang_thai[1])
    torch.set_rng_state(trang_thai[2])
    return pd.DataFrame(dong), nhat_ky_khong_dung


# ==============================================================================================================
# ve_early_stopping  (def)
# Vị trí: Cell 53 · mục 8.E Chạy 5 run ghép cặp và xuất Tables A, B
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 8.2b: loss TRAIN/VAL của bản H1 không dừng sớm qua 150 epoch, vạch dọc ở epoch bản có early
#            stopping dừng, chấm ở trọng số được giữ.
# ==============================================================================================================
def ve_early_stopping(run_artifacts: Dict[int, Dict[str, Any]], nhat_ky_khong_dung: Dict[int, Dict[str, Any]],
                      duong_dan_luu: Path) -> None:
    """
    Hàm này để làm gì: Vẽ loss TRAIN/VAL của bản H1 KHÔNG early stopping trong đủ 150 epoch, kèm vạch dọc tại epoch
    mà bản CÓ early stopping đã dừng và chấm tại epoch có loss VAL thấp nhất, để thấy vì sao dừng sớm.

    Nhận vào:
        run_artifacts (Dict[int, Dict[str, Any]]): Kết quả 5 run (nhật ký bản có early stopping).
        nhat_ky_khong_dung (Dict[int, Dict[str, Any]]): Nhật ký bản không early stopping.
        duong_dan_luu (Path): Đường dẫn lưu PNG.
    """
    cac_seed = list(nhat_ky_khong_dung.keys())
    fig, axes = plt.subplots(1, len(cac_seed), figsize=(4 * len(cac_seed), 3.8), sharey=True)
    for ax, seed in zip(axes, cac_seed):
        log_no = nhat_ky_khong_dung[seed]
        log_es = run_artifacts[seed]['H1_log']
        ep = np.arange(1, len(log_no['train_loss']) + 1)
        ax.plot(ep, log_no['train_loss'], color='#2b6cb0', lw=1.2, label='TRAIN')
        ax.plot(ep, log_no['val_loss'], color='#c05621', lw=1.2, label='VAL')
        ax.axvline(log_es['epochs_run'], color='#4a5568', ls='--', lw=1.1, label='early stopping dừng')
        ax.scatter([log_es['best_epoch'] + 1], [log_es['best_val_bce']], color='#c53030', s=40, zorder=5,
                   label='trọng số được giữ')
        ax.set_title(f'seed {seed}', fontsize=10)
        ax.set_xlabel('Epoch')
    axes[0].set_ylabel('BCE loss')
    axes[0].legend(fontsize=7)
    fig.suptitle('Hình 8.2b. H1 khi KHÔNG early stopping (đủ 150 epoch): loss VAL không giảm thêm sau vài epoch đầu', fontsize=11)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()
