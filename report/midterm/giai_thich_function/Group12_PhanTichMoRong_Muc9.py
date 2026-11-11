# -*- coding: utf-8 -*-
"""
NHÓM 12 — Phân tích mở rộng: kết cục khác, AIPW, trung gian, CATE, độ nhạy RV (mục 9)

Mục 9 viết dạng tuần tự (không bọc hàm) cho 9.B–9.G; ở file này mỗi đoạn được bọc thành một hàm chỉ để đọc.
9.H có hai hàm thật: tính Robustness Value và partial R^2 làm mốc.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - AdvancedConfig                             Cell 69 (class)
  - doan_ma_cell_69_nap_hai_quan_the           Cell 69 (đoạn mã)
  - doan_ma_cell_71_diem_hk2_ols               Cell 71 (đoạn mã)
  - doan_ma_cell_73_ty_le_dat_tin_chi_ols      Cell 73 (đoạn mã)
  - doan_ma_cell_75_giu_chan_aipw              Cell 75 (đoạn mã)
  - doan_ma_cell_77_trung_gian_nde_nie         Cell 77 (đoạn mã)
  - doan_ma_cell_79_cate_theo_nhom             Cell 79 (đoạn mã)
  - KetQuaDoNhayCinelli                        Cell 83 (class)
  - tinh_rv_cinelli_hazlett                    Cell 83 (def)
  - benchmark_partial_r2                       Cell 83 (def)
"""


# ==============================================================================================================
# AdvancedConfig  (class)
# Vị trí: Cell 69 · mục 9.A Hai quần thể phân tích cho phần mở rộng
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Cấu hình riêng cho mục 9: đường dẫn dữ liệu, seed 42, 1000 bootstrap và 7 hiệp biến (thêm
#            Displaced, Gender so với mục 8).
# LƯU Ý    : Mục 9 dùng biến gốc (tuổi, điểm liên tục) và 7 hiệp biến, khác mã hóa rời rạc 5 biến của mục 8.
# ==============================================================================================================
@dataclass(frozen=True)
class AdvancedConfig:
    data_path: str = 'data/uci_697.csv'
    artifacts_dir: str = 'artifacts'
    tables_dir: str = 'artifacts/tables'
    seed: int = 42
    n_bootstrap: int = 1000
    confounders: Tuple[str, ...] = (
        'Age at enrollment',
        'Admission grade',
        "Mother's qualification",
        'Debtor',
        'Tuition fees up to date',
        'Displaced',
        'Gender'
    )


# ==============================================================================================================
# doan_ma_cell_69_nap_hai_quan_the  (đoạn mã tuần tự)
# Vị trí: Cell 69 · mục 9.A Hai quần thể phân tích cho phần mở rộng
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.A: tạo hai quần thể: df_completed (3,630 SV có kết cục) và df_full (4,424 SV, gồm Enrolled)
#            với kết cục giữ chân Y_retain = 1 nếu không bỏ học.
# ❓ HỎI   : Vì sao cần quần thể 4,424?
#   ➜ ĐÁP : Để kiểm tra kết luận không phụ thuộc việc loại Enrolled (mục 9.D).
# ==============================================================================================================
def doan_ma_cell_69_nap_hai_quan_the():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    ADV_CONFIG = AdvancedConfig()
    Path(ADV_CONFIG.tables_dir).mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(ADV_CONFIG.data_path, sep=None, engine='python')
    print(f'[Nạp dữ liệu] Tổng số dòng ban đầu: {len(df_raw):,} dòng')
    print(f'[Phân bố Target ban đầu]: {df_raw["Target"].value_counts().to_dict()}')
    df_completed = df_raw[df_raw['Target'] != 'Enrolled'].copy().reset_index(drop=True)
    df_completed['T'] = df_completed['Scholarship holder'].astype(int)
    df_completed['Y_grad'] = (df_completed['Target'] == 'Graduate').astype(int)
    df_completed['Z1_age'] = df_completed['Age at enrollment'].astype(float)
    df_completed['Z2_adm'] = df_completed['Admission grade'].astype(float)
    df_completed['Z3_mom'] = (df_completed["Mother's qualification"] > 1).astype(int)
    df_completed['Z4_debt'] = df_completed['Debtor'].astype(int)
    df_completed['Z5_fee'] = df_completed['Tuition fees up to date'].astype(int)
    df_completed['Z6_disp'] = df_completed['Displaced'].astype(int)
    df_completed['Z7_gen'] = df_completed['Gender'].astype(int)
    df_full = df_raw.copy().reset_index(drop=True)
    df_full['T'] = df_full['Scholarship holder'].astype(int)
    df_full['Y_retain'] = (df_full['Target'] != 'Dropout').astype(int)
    df_full['Z1_age'] = df_full['Age at enrollment'].astype(float)
    df_full['Z2_adm'] = df_full['Admission grade'].astype(float)
    df_full['Z3_mom'] = (df_full["Mother's qualification"] > 1).astype(int)
    df_full['Z4_debt'] = df_full['Debtor'].astype(int)
    df_full['Z5_fee'] = df_full['Tuition fees up to date'].astype(int)
    df_full['Z6_disp'] = df_full['Displaced'].astype(int)
    df_full['Z7_gen'] = df_full['Gender'].astype(int)
    COV_COLS = ['Z1_age', 'Z2_adm', 'Z3_mom', 'Z4_debt', 'Z5_fee', 'Z6_disp', 'Z7_gen']
    print(f'[Sanity Check] Cohort Hoàn tất: {len(df_completed)} dòng | Cohort Toàn trường: {len(df_full)} dòng.')


# ==============================================================================================================
# doan_ma_cell_71_diem_hk2_ols  (đoạn mã tuần tự)
# Vị trí: Cell 71 · mục 9.B Kết cục 1 — điểm trung bình HK2 ($Y_{\text{grade}}$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.B: hiệu ứng của học bổng lên điểm trung bình HK2 bằng hồi quy OLS Y_grade ~ T + 7 hiệp biến.
# CÁCH LÀM :
#   1. Hệ số của T là ATE (thang 20): +1.48, CI [1.09, 1.86].
#   2. So với chênh lệch thô +2.63: điều chỉnh làm giảm gần một nửa.
# LƯU Ý    : Điểm HK2 là kết cục sau T, ở đây dùng làm KẾT CỤC (không phải biến điều chỉnh) nên hợp lệ.
# ==============================================================================================================
def doan_ma_cell_71_diem_hk2_ols():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    df_completed['Y_grade'] = df_completed['Curricular units 2nd sem (grade)'].astype(float)
    X_grade = sm.add_constant(df_completed[['T'] + COV_COLS])
    model_grade = sm.OLS(df_completed['Y_grade'], X_grade).fit()
    ate_grade = model_grade.params['T']
    ci_grade_low, ci_grade_high = model_grade.conf_int().loc['T']
    p_grade = model_grade.pvalues['T']
    r2_grade = model_grade.rsquared
    print('=== KẾT QUẢ ĐO LƯỜNG CAN THIỆP TRÊN ĐIỂM SỐ HK2 (GPA) ===')
    print(f'ATE (Điểm GPA tăng thêm): +{ate_grade:.3f} điểm (trên thang 20)')
    print(f'Khoảng tin cậy 95% CI: [{ci_grade_low:.3f}, {ci_grade_high:.3f}]')
    print(f'p-value: {p_grade:.2e} (Cực kỳ có ý nghĩa thống kê)')
    print(f'R-squared của mô hình: {r2_grade:.4f}')
    mean_t0 = df_completed[df_completed['T'] == 0]['Y_grade'].mean()
    mean_t1 = df_completed[df_completed['T'] == 1]['Y_grade'].mean()
    print(f'Điểm trung bình quan sát: Không học bổng = {mean_t0:.2f} | Có học bổng = {mean_t1:.2f} | Chênh lệch thô = {mean_t1 - mean_t0:+.2f}')


# ==============================================================================================================
# doan_ma_cell_73_ty_le_dat_tin_chi_ols  (đoạn mã tuần tự)
# Vị trí: Cell 73 · mục 9.C Kết cục 2 — tỷ lệ đỗ tín chỉ HK2 ($Y_{\text{velocity}}$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.C: hiệu ứng lên tỷ lệ đạt tín chỉ HK2 = số tín chỉ đạt / số đăng ký (mẫu số tối thiểu 1 để
#            tránh chia 0), OLS với 7 hiệp biến.
# CÁCH LÀM :
#   1. ATE = +14.5 điểm %, CI [11.8%, 17.2%].
# ==============================================================================================================
def doan_ma_cell_73_ty_le_dat_tin_chi_ols():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    enrolled_credits = np.maximum(df_completed['Curricular units 2nd sem (enrolled)'].values, 1.0)
    df_completed['Y_velocity'] = df_completed['Curricular units 2nd sem (approved)'].values / enrolled_credits
    X_vel = sm.add_constant(df_completed[['T'] + COV_COLS])
    model_vel = sm.OLS(df_completed['Y_velocity'], X_vel).fit()
    ate_vel = model_vel.params['T']
    ci_vel_low, ci_vel_high = model_vel.conf_int().loc['T']
    p_vel = model_vel.pvalues['T']
    print('=== KẾT QUẢ ĐO LƯỜNG CAN THIỆP TRÊN TỐC ĐỘ TÍN CHỈ ===')
    print(f'ATE (Tỷ lệ đỗ tín chỉ tăng thêm): +{ate_vel*100:.2f}%')
    print(f'Khoảng tin cậy 95% CI: [+{ci_vel_low*100:.2f}%, +{ci_vel_high*100:.2f}%]')
    print(f'p-value: {p_vel:.2e}')
    full_pass_t0 = np.mean(df_completed[df_completed['T'] == 0]['Y_velocity'] == 1.0)
    full_pass_t1 = np.mean(df_completed[df_completed['T'] == 1]['Y_velocity'] == 1.0)
    print(f'Tỷ lệ đỗ 100% số môn đăng ký: Nhóm T=0: {full_pass_t0*100:.1f}% | Nhóm T=1: {full_pass_t1*100:.1f}%')


# ==============================================================================================================
# doan_ma_cell_75_giu_chan_aipw  (đoạn mã tuần tự)
# Vị trí: Cell 75 · mục 9.D Kết cục 3 — giữ chân toàn trường ($Y_{\text{retain}}, N = 4,424$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.D: hiệu ứng lên tỷ lệ giữ chân trên 4,424 SV bằng AIPW (doubly robust).
# CÁCH LÀM :
#   1. Mô hình propensity: logistic T ~ hiệp biến, chặn e(x) trong [0.05, 0.95] để tránh trọng số cực lớn.
#   2. Hai mô hình kết cục: logistic riêng cho nhóm T=0 (mu0) và T=1 (mu1).
#   3. Điểm AIPW: dr1 = mu1 + T(Y - mu1)/e; dr0 = mu0 + (1-T)(Y - mu0)/(1-e).
#   4. ATE = trung bình(dr1) - trung bình(dr0) = +16.87%.
#   5. CI: bootstrap 1000 lần trên các điểm dr (không khớp lại mô hình) -> [13.79%, 19.73%].
# LÝ THUYẾT: Doubly robust: ước lượng nhất quán nếu MỘT TRONG HAI mô hình (propensity hoặc kết cục) đúng.
# ❓ HỎI   : Vì sao +16.87% nhỏ hơn +24%?
#   ➜ ĐÁP : Kết cục khác (giữ chân gồm cả đang học) và quần thể khác; điều quan trọng là vẫn dương rõ.
# ==============================================================================================================
def doan_ma_cell_75_giu_chan_aipw():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    ps_model = LogisticRegression(max_iter=500, random_state=ADV_CONFIG.seed)  # mô hình xác suất được học bổng e(x)
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
    print(f'Khoảng tin cậy 95% Bootstrap: [+{ci_ret_low*100:.2f}%, +{ci_ret_high*100:.2f}%]')


# ==============================================================================================================
# doan_ma_cell_77_trung_gian_nde_nie  (đoạn mã tuần tự)
# Vị trí: Cell 77 · mục 9.E Phân tích trung gian (NDE / NIE)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.E: tách hiệu ứng thành trực tiếp (NDE) và gián tiếp qua tín chỉ HK1 (NIE) bằng phương pháp
#            tích hệ số.
# CÁCH LÀM :
#   1. Mô hình M ~ T + hiệp biến: hệ số a của T (+0.73 tín chỉ).
#   2. Mô hình Y ~ T + M + hiệp biến (LPM): NDE = hệ số của T, b = hệ số của M.
#   3. NIE = a x b; tổng = NDE + NIE; tỷ lệ trung gian = NIE / tổng = 21.5%.
# LÝ THUYẾT: Giả định thêm: không có nhiễu giữa M và Y, và mô hình tuyến tính không có tương tác T x M.
# ❓ HỎI   : 21.5% nghĩa là gì?
#   ➜ ĐÁP : Khoảng một phần năm tác động của học bổng đi qua việc học tốt hơn ở HK1; phần lớn còn lại là trực
#           tiếp (tài chính, động lực...).
# ==============================================================================================================
def doan_ma_cell_77_trung_gian_nde_nie():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    df_med = df_completed.copy()  # 3,630 SV có kết cục
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
    print(f'==> TỶ LỆ TRUNG GIAN QUA ĐÀ HỌC TẬP HK1: {prop_mediated:.1f}%')


# ==============================================================================================================
# doan_ma_cell_79_cate_theo_nhom  (đoạn mã tuần tự)
# Vị trí: Cell 79 · mục 9.F Hiệu ứng theo nhóm (CATE)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mục 9.F: CATE theo 6 nhóm định trước (tuổi, Debtor, nơi ở) bằng G-computation logistic khớp riêng
#            trong từng nhóm.
# CÁCH LÀM :
#   1. Trong mỗi nhóm: logistic Y ~ T + hiệp biến (bỏ hiệp biến hằng trong nhóm).
#   2. Đặt T=0 cho mọi người, dự đoán, lấy trung bình -> Risk(do(0)); tương tự với T=1.
#   3. CATE = hiệu; cột "x" là tỷ số rủi ro Risk(1)/Risk(0).
# LƯU Ý    : Chưa có CI cho từng nhóm nên chênh lệch giữa các nhóm cần đọc thận trọng; "2.03x" là tỷ số rủi ro,
#            không phải "hưởng lợi gấp đôi".
# ==============================================================================================================
def doan_ma_cell_79_cate_theo_nhom():
    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm
    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)
    subgroups_specs = [  # 6 nhóm ĐỊNH TRƯỚC: (tên, điều kiện lọc)
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
    display(df_cate_display)


# ==============================================================================================================
# KetQuaDoNhayCinelli  (class)
# Vị trí: Cell 83 · mục 9.H Độ nhạy trước nhiễu ẩn — Robustness Value (Cinelli & Hazlett, 2020)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass kết quả độ nhạy: hệ số của T, t-stat, bậc tự do, partial R^2 của T với Y, RV và RV ở mức
#            alpha 0.05.
# ==============================================================================================================
@dataclass
class KetQuaDoNhayCinelli:
    he_so_T: float
    t_stat: float
    bac_tu_do: int
    r2_partial_YT: float
    rv_q1: float
    rv_q1_alpha05: float


# ==============================================================================================================
# tinh_rv_cinelli_hazlett  (def)
# Vị trí: Cell 83 · mục 9.H Độ nhạy trước nhiễu ẩn — Robustness Value (Cinelli & Hazlett, 2020)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính Robustness Value (Cinelli & Hazlett 2020) từ hồi quy OLS Y ~ T + Z1..Z5: một nhiễu ẩn phải
#            mạnh cỡ nào mới xóa được hiệu ứng.
# CÁCH LÀM :
#   1. Mã hóa 5 Z (thứ tự riêng của mục 9.H: Z1 tuổi, Z2 điểm vào, Z3 học vấn mẹ, Z4 Debtor, Z5 học phí).
#   2. Giải bình phương tối thiểu, tính sai số chuẩn và t-stat của T (t = 14.20, df = 3623).
#   3. f = |t| / sqrt(df); RV = 0.5 x (sqrt(f^4 + 4 f^2) - f^2) = 20.97%.
#   4. RV ở alpha 0.05: thay f bằng f - t*_{0.975, df-1}/sqrt(df-1) -> 18.37%.
#   5. Partial R^2 của T với Y = t^2 / (t^2 + df) = 5.27%.
# LÝ THUYẾT: RV = 21% nghĩa là nhiễu ẩn phải giải thích ít nhất khoảng 21% phương sai dư của CẢ T lẫn Y thì mới
#            đưa ước lượng về 0.
# ❓ HỎI   : 21% là lớn hay nhỏ?
#   ➜ ĐÁP : So với biến quan sát mạnh nhất là học phí (partial R^2 với Y 10.4%): kết luận vững trước nhiễu ẩn
#           yếu hơn học phí, nhưng không miễn nhiễm với nhiễu mạnh gấp đôi.
# ==============================================================================================================
def tinh_rv_cinelli_hazlett(du_lieu: pd.DataFrame) -> KetQuaDoNhayCinelli:
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


# ==============================================================================================================
# benchmark_partial_r2  (def)
# Vị trí: Cell 83 · mục 9.H Độ nhạy trước nhiễu ẩn — Robustness Value (Cinelli & Hazlett, 2020)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Partial R^2 của từng biến nền với Y (khi đã có T và các biến nền khác), làm mốc so sánh cho RV.
# CÁCH LÀM :
#   1. RSS đầy đủ: hồi quy Y ~ T + mọi Z.
#   2. Với mỗi Z: RSS khi bỏ Z đó.
#   3. Partial R^2 = (RSS bỏ Z - RSS đầy đủ) / RSS bỏ Z.
# ==============================================================================================================
def benchmark_partial_r2(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Partial R^2 của từng biến nền quan sát được với Y (sau khi đã có T và các biến nền còn lại),
    dùng làm mốc so sánh cho Robustness Value.

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        pd.DataFrame: Tên biến, mô tả, partial R^2.
    """
    Z = pd.DataFrame({
        'Z1': pd.qcut(du_lieu['Age_at_enrollment'], q=3, labels=[0, 1, 2]).astype(int),
        'Z2': (du_lieu['Admission_grade'] >= du_lieu['Admission_grade'].median()).astype(int),
        'Z3': (du_lieu['Mothers_qualification'] > 1).astype(int),
        'Z4': du_lieu['Debtor'].astype(int),
        'Z5': du_lieu['Tuition_fees_up_to_date'].astype(int),
    })
    mo_ta = {'Z1': 'Tuổi nhập học (3 nhóm)', 'Z2': 'Điểm vào >= trung vị', 'Z3': 'Học vấn mẹ > 1',
             'Z4': 'Debtor', 'Z5': 'Học phí đúng hạn'}
    y = du_lieu['Target_Graduate'].values.astype(float)
    t = du_lieu['Scholarship_holder'].values.astype(float)

    def rss(cot: List[str]) -> float:
        X = np.column_stack([np.ones(len(y)), t] + [Z[c].values for c in cot]).astype(float)
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return float(r @ r)

    day_du = rss(list(Z.columns))
    dong = []
    for c in Z.columns:
        thieu = rss([k for k in Z.columns if k != c])
        dong.append({'Confounder': c, 'Mô tả': mo_ta[c], 'Partial_R2_with_Outcome': (thieu - day_du) / thieu})
    return pd.DataFrame(dong)
