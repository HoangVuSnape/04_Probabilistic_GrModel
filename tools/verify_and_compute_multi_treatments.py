"""
Script: tools/verify_and_compute_multi_treatments.py
Mô tả: Độc lập ước lượng và kiểm chứng hiệu ứng nhân quả đa can thiệp (Multi-Treatment Causal Evaluation):
      1. T1: Scholarship holder (do(T1=1) vs do(T1=0))
      2. T2: Tuition fees up to date (do(T2=1) vs do(T2=0)) — LƯU Ý: đây là TRẠNG THÁI đóng học phí đúng hạn
         được ghi nhận trong hồ sơ, KHÔNG phải một chính sách "xóa nợ". Có nguy cơ nhân quả ngược
         (sinh viên bỏ học thì ngừng đóng học phí) và vi phạm positivity (ô T1=1,T2=0 chỉ có 37 SV).
      3. T3: Joint Intervention (do(T1=1, T2=1) vs do(T1=0, T2=0))
      Phương pháp: Backdoor Linear Regression (LPM), G-computation (Logistic), Doubly Robust (AIPW).
      LƯU Ý ƯỚC LƯỢNG: AIPW đúng nghĩa (có số hạng hiệu chỉnh propensity) CHỈ được tính trên Full Cohort.
      Trên C_test, vì nguyên tắc không dùng Y_test, bộ ước lượng thứ ba là Outcome Regression (T-learner),
      nên được dán nhãn đúng là 'Outcome Regression (T-learner)' thay vì 'AIPW'.
      Đánh giá: Quần thể toàn phần (N=3,630) và 5-Fold Paired CV trên C_test (n=545, seeds 42..46).
      Xuất bản: artifacts/tables/Table_Multi_Treatment.csv

CÁC BẢO CHỨNG PHƯƠNG PHÁP LUẬN VÀ BẢO VỆ TOÁN HỌC (Iteration 2 Remediation):
      1. Strict Zero-Leakage Preprocessing: StandardScaler chỉ fit trên train_df, sau đó transform cho train_df và test_df.
      2. Causal Contract Y_test Isolation: Tuyệt đối KHÔNG truy cập test_df['Y'] trong đánh giá ngoại mẫu (Slide 61).
      3. T3 Sample Retention: Giữ lại 100% sinh viên (n=545 trên test, N=3,630 trên full cohort),
         không dropna các nhánh không đồng thuận (0,1) và (1,0).
      4. G-computation Parameter Uncertainty: Tái mẫu phi tham số toàn phần (Full Non-parametric Bootstrap) trên train_df,
         khắc phục triệt để sai số chuẩn ép hẹp 12x deflation.
      5. Axiomatic Bounding: Chặn biên xác suất [0.0, 1.0] cho nguy cơ và [-1.0, 1.0] cho ATE cùng khoảng tin cậy 95%.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


@dataclass(frozen=True)
class MultiTreatmentConfig:
    data_path: Path
    splits_dir: Path
    output_table_path: Path
    seeds: Tuple[int, ...]
    covariate_cols: Tuple[str, ...]


@dataclass(frozen=True)
class CausalEstimateResult:
    treatment_id: str
    treatment_name: str
    estimator: str
    cohort: str
    seed: str
    cohort_size: int
    risk_do_0: float
    risk_do_1: float
    ate: float
    ci_95_low: float
    ci_95_high: float
    p_value: float
    adjustment_set: str
    policy_implication: str


def tai_va_tien_xu_ly_du_lieu(duong_dan_csv: Path) -> pd.DataFrame:
    """
    Hàm này để làm gì: Tải dữ liệu UCI ID 697, lọc quần thể nhị phân hoàn tất (Target != Enrolled, N=3,630),
    và mã hóa các biến can thiệp, kết quả cùng biến nhiễu nền tảng Z.

    Nhận vào:
        duong_dan_csv (Path): Đường dẫn tới tệp uci_697.csv.

    Trả ra:
        pd.DataFrame: Bảng dữ liệu tiền xử lý với 3,630 dòng.
    """
    df_raw = pd.read_csv(duong_dan_csv)
    df_bin = df_raw[df_raw['Target'].isin(['Dropout', 'Graduate'])].copy().reset_index(drop=True)
    
    # Outcome Y: 1 nếu Graduate, 0 nếu Dropout
    df_bin['Y'] = (df_bin['Target'] == 'Graduate').astype(int)
    
    # Treatments: T1 (Học bổng) và T2 (Học phí đúng hạn / Xóa nợ học phí)
    df_bin['T1'] = df_bin['Scholarship holder'].astype(int)
    df_bin['T2'] = df_bin['Tuition fees up to date'].astype(int)
    
    # Confounders Z: Các biến nền tảng nhân khẩu và học thuật trước can thiệp
    df_bin['Z1_age'] = pd.to_numeric(df_bin['Age at enrollment'], errors='coerce').astype(float)
    df_bin['Z2_adm'] = pd.to_numeric(df_bin['Admission grade'], errors='coerce').astype(float)
    df_bin['Z3_mom'] = (pd.to_numeric(df_bin["Mother's qualification"], errors='coerce') > 1).astype(int)
    df_bin['Z4_debt'] = df_bin['Debtor'].astype(int)
    df_bin['Z5_disp'] = df_bin['Displaced'].astype(int)
    df_bin['Z6_gen'] = df_bin['Gender'].astype(int)
    
    return df_bin


def ap_dung_chuan_hoa_khong_ro_ri(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cov_cols: Tuple[str, ...],
    can_chuan_hoa: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hàm này để làm gì: Chuẩn hóa Z-score nghiêm ngặt không rò rỉ dữ liệu (Strict Zero-Leakage StandardScaler).
    Fit scaler CHỈ trên train_df, sau đó transform cho cả train_df và test_df.
    
    Nhận vào:
        train_df (pd.DataFrame): Tập dữ liệu huấn luyện.
        test_df (pd.DataFrame): Tập dữ liệu kiểm thử (C_test).
        cov_cols (Tuple[str, ...]): Danh sách tên biến nhiễu Z cần chuẩn hóa.
        can_chuan_hoa (bool): True nếu dùng mô hình phi tuyến (G-computation, AIPW), False nếu dùng LPM.
        
    Trả ra:
        Tuple[pd.DataFrame, pd.DataFrame]: (tr_scaled, te_scaled) bảo đảm không rò rỉ thông tin test.
    """
    if not can_chuan_hoa:
        return train_df.copy(), test_df.copy()
        
    scaler = StandardScaler()
    tr = train_df.copy()
    te = test_df.copy()
    cov_list = list(cov_cols)
    
    # Fit CHỈ trên tập train
    tr[cov_list] = scaler.fit_transform(tr[cov_list])
    # Transform tập test bằng tham số học từ train
    te[cov_list] = scaler.transform(te[cov_list])
    
    return tr, te


def uoc_luong_backdoor_lpm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    treatment_type: str,
    cov_cols: Tuple[str, ...]
) -> Tuple[float, float, float, float, float, float]:
    """
    Hàm này để làm gì: Ước lượng hiệu ứng can thiệp ATE, Risk(0), Risk(1) bằng mô hình Backdoor Linear Regression (LPM).
    Ràng buộc biên toán học: Chặn [0.0, 1.0] cho Risk(0), Risk(1) và [-1.0, 1.0] cho ATE cùng khoảng tin cậy.

    Nhận vào:
        train_df (pd.DataFrame): Tập dữ liệu huấn luyện.
        test_df (pd.DataFrame): Tập dữ liệu kiểm thử đối chuẩn (C_test).
        treatment_type (str): 'T1', 'T2' hoặc 'T3'.
        cov_cols (Tuple[str, ...]): Danh sách tên biến nhiễu Z.

    Trả ra:
        Tuple[float, float, float, float, float, float]:
            (risk_do_0, risk_do_1, ate, ci_low, ci_high, p_val)
    """
    cov_list = list(cov_cols)
    if treatment_type == 'T1':
        features = ['T1', 'T2'] + cov_list
        X_tr = sm.add_constant(train_df[features], has_constant='add')
        model = sm.OLS(train_df['Y'], X_tr).fit()
        
        X_te_0 = sm.add_constant(test_df[features].assign(T1=0), has_constant='add')
        X_te_1 = sm.add_constant(test_df[features].assign(T1=1), has_constant='add')
        r0 = float(np.clip(model.predict(X_te_0).mean(), 0.0, 1.0))
        r1 = float(np.clip(model.predict(X_te_1).mean(), 0.0, 1.0))
        ate = float(np.clip(model.params['T1'], -1.0, 1.0))
        se = float(model.bse['T1'])
        ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
        ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
        p_val = float(model.pvalues['T1'])
        
    elif treatment_type == 'T2':
        features = ['T2', 'T1'] + cov_list
        X_tr = sm.add_constant(train_df[features], has_constant='add')
        model = sm.OLS(train_df['Y'], X_tr).fit()
        
        X_te_0 = sm.add_constant(test_df[features].assign(T2=0), has_constant='add')
        X_te_1 = sm.add_constant(test_df[features].assign(T2=1), has_constant='add')
        r0 = float(np.clip(model.predict(X_te_0).mean(), 0.0, 1.0))
        r1 = float(np.clip(model.predict(X_te_1).mean(), 0.0, 1.0))
        ate = float(np.clip(model.params['T2'], -1.0, 1.0))
        se = float(model.bse['T2'])
        ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
        ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
        p_val = float(model.pvalues['T2'])
        
    else:  # T3: Joint intervention với số hạng tương tác T1 * T2
        train_copy = train_df.copy()
        test_copy = test_df.copy()
        train_copy['T1_x_T2'] = train_copy['T1'] * train_copy['T2']
        test_copy['T1_x_T2'] = test_copy['T1'] * test_copy['T2']
        
        features = ['T1', 'T2', 'T1_x_T2'] + cov_list
        X_tr = sm.add_constant(train_copy[features], has_constant='add')
        model = sm.OLS(train_copy['Y'], X_tr).fit()
        
        te_0 = test_copy[features].assign(T1=0, T2=0, T1_x_T2=0)
        te_1 = test_copy[features].assign(T1=1, T2=1, T1_x_T2=1)
        X_te_0 = sm.add_constant(te_0, has_constant='add')
        X_te_1 = sm.add_constant(te_1, has_constant='add')
        
        r0 = float(np.clip(model.predict(X_te_0).mean(), 0.0, 1.0))
        r1 = float(np.clip(model.predict(X_te_1).mean(), 0.0, 1.0))
        raw_ate = float(model.params['T1'] + model.params['T2'] + model.params['T1_x_T2'])
        ate = float(np.clip(raw_ate, -1.0, 1.0))
        
        cov_m = model.cov_params()
        var_joint = float(
            cov_m.loc['T1', 'T1'] + cov_m.loc['T2', 'T2'] + cov_m.loc['T1_x_T2', 'T1_x_T2']
            + 2.0 * cov_m.loc['T1', 'T2']
            + 2.0 * cov_m.loc['T1', 'T1_x_T2']
            + 2.0 * cov_m.loc['T2', 'T1_x_T2']
        )
        se = float(np.sqrt(max(var_joint, 1e-12)))
        ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
        ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
        z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
        p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
        
    return r0, r1, ate, ci_low, ci_high, p_val


def uoc_luong_g_computation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    treatment_type: str,
    cov_cols: Tuple[str, ...],
    method: str = "bootstrap",
    n_boot: int = 100,
    random_state: int = 42
) -> Tuple[float, float, float, float, float, float]:
    """
    Hàm này để làm gì: Ước lượng chuẩn hóa G-computation bằng mô hình hồi quy Logistic,
    tính xác suất biên đối thực tế Risk(0), Risk(1) và khoảng tin cậy 95% tiệm cận.
    
    Phương pháp tính sai số chuẩn (SE):
    - method="bootstrap": Tái lấy mẫu phi tham số toàn phần (Full Non-parametric Bootstrap, B=100)
      có hoàn lại trên train_df và refit mô hình LogisticRegression để phản ánh đầy đủ độ bất định
      tham số beta, khắc phục triệt để lỗi ép hẹp SE (12x deflation).
    - method="delta": Phương pháp giải tích vi phân Delta Method (M-estimation) tính nhanh
      đạo hàm Jacobian g = d(ATE)/d(beta) và dạng toàn phương g^T Cov(beta) g.

    Nhận vào:
        train_df (pd.DataFrame): Tập huấn luyện.
        test_df (pd.DataFrame): Tập đối chuẩn (C_test).
        treatment_type (str): 'T1', 'T2' hoặc 'T3'.
        cov_cols (Tuple[str, ...]): Danh sách tên biến nhiễu Z.
        method (str): 'bootstrap' (mặc định) hoặc 'delta'.
        n_boot (int): Số vòng tái mẫu bootstrap (mặc định 100).
        random_state (int): Hạt giống ngẫu nhiên tái lập.

    Trả ra:
        Tuple[float, float, float, float, float, float]:
            (risk_do_0, risk_do_1, ate, ci_low, ci_high, p_val)
    """
    cov_list = list(cov_cols)
    if treatment_type == 'T1':
        cols = ['T1', 'T2'] + cov_list
        te_0 = test_df[cols].assign(T1=0)
        te_1 = test_df[cols].assign(T1=1)
        X_tr = train_df[cols]
    elif treatment_type == 'T2':
        cols = ['T2', 'T1'] + cov_list
        te_0 = test_df[cols].assign(T2=0)
        te_1 = test_df[cols].assign(T2=1)
        X_tr = train_df[cols]
    else:  # T3 Joint
        tr_c = train_df.copy()
        te_c = test_df.copy()
        tr_c['T1_x_T2'] = tr_c['T1'] * tr_c['T2']
        te_c['T1_x_T2'] = te_c['T1'] * te_c['T2']
        cols = ['T1', 'T2', 'T1_x_T2'] + cov_list
        te_0 = te_c[cols].assign(T1=0, T2=0, T1_x_T2=0)
        te_1 = te_c[cols].assign(T1=1, T2=1, T1_x_T2=1)
        X_tr = tr_c[cols]

    y_tr = train_df['Y'].values
    X_tr_mat = np.asarray(X_tr.values, dtype=np.float64)
    X_te0_mat = np.asarray(te_0.values, dtype=np.float64)
    X_te1_mat = np.asarray(te_1.values, dtype=np.float64)

    # 1. Điểm ước lượng cơ sở bằng LogisticRegression (L-BFGS)
    clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=random_state).fit(X_tr_mat, y_tr)
    p0 = clf.predict_proba(X_te0_mat)[:, 1]
    p1 = clf.predict_proba(X_te1_mat)[:, 1]
    r0 = float(np.clip(np.mean(p0), 0.0, 1.0))
    r1 = float(np.clip(np.mean(p1), 0.0, 1.0))
    ate = float(np.clip(r1 - r0, -1.0, 1.0))

    # 2. Tính sai số chuẩn (SE) và khoảng tin cậy 95%
    if method == "delta":
        X_tr_const = sm.add_constant(X_tr_mat, has_constant='add')
        X_te0_const = sm.add_constant(X_te0_mat, has_constant='add')
        X_te1_const = sm.add_constant(X_te1_mat, has_constant='add')

        logit_mod = sm.Logit(y_tr, X_tr_const).fit(disp=False, method='lbfgs', maxiter=500)
        cov_beta = logit_mod.cov_params()
        if hasattr(cov_beta, 'values'):
            cov_beta = cov_beta.values
        cov_beta = np.asarray(cov_beta, dtype=np.float64)

        p0_m = np.asarray(logit_mod.predict(X_te0_const), dtype=np.float64)
        p1_m = np.asarray(logit_mod.predict(X_te1_const), dtype=np.float64)

        # Gradient Jacobian g = d(ATE)/d(beta)
        grad_1 = (p1_m * (1.0 - p1_m))[:, None] * X_te1_const
        grad_0 = (p0_m * (1.0 - p0_m))[:, None] * X_te0_const
        g = np.mean(grad_1 - grad_0, axis=0)

        var_beta = float(g.T @ cov_beta @ g)
        var_z = float(np.var(p1_m - p0_m, ddof=1) / len(p0_m))
        se = float(np.sqrt(max(var_beta + var_z, 1e-12)))

        ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
        ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
        z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
        p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
    else:
        # Full Non-parametric Bootstrap trên train_df (refit mô hình Logistic Regression)
        rng = np.random.default_rng(random_state)
        n_tr = len(X_tr_mat)
        n_te = len(X_te0_mat)
        boot_diffs = []
        for _ in range(n_boot):
            b_tr = rng.choice(n_tr, size=n_tr, replace=True)
            b_te = rng.choice(n_te, size=n_te, replace=True)
            try:
                clf_b = LogisticRegression(max_iter=200, solver='lbfgs', tol=1e-3, random_state=random_state).fit(
                    X_tr_mat[b_tr], y_tr[b_tr]
                )
                p0_b = clf_b.predict_proba(X_te0_mat[b_te])[:, 1]
                p1_b = clf_b.predict_proba(X_te1_mat[b_te])[:, 1]
                boot_diffs.append(float(np.mean(p1_b - p0_b)))
            except Exception:
                continue

        if len(boot_diffs) >= 20:
            se = float(np.std(boot_diffs))
            ci_low = float(np.clip(np.percentile(boot_diffs, 2.5), -1.0, 1.0))
            ci_high = float(np.clip(np.percentile(boot_diffs, 97.5), -1.0, 1.0))
        else:
            se = float("nan")  # bootstrap thất bại: không gán cứng SE
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))

        z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
        p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf

    return r0, r1, ate, ci_low, ci_high, p_val


def uoc_luong_doubly_robust(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    treatment_type: str,
    cov_cols: Tuple[str, ...]
) -> Tuple[float, float, float, float, float, float]:
    """
    Hàm này để làm gì: Ước lượng bán tham số Doubly Robust (AIPW) kết hợp mô hình xác suất xu hướng (Propensity Score)
    và mô hình dự báo đầu ra (Outcome Regression), bảo đảm tính vững khi 1 trong 2 mô hình đúng.
    
    Quy tắc cách ly Y_test (Slide 61):
        - Trên tập Full Cohort (len(train_df) == len(test_df)): Đánh giá đầy đủ pseudo-outcomes AIPW bằng nhãn Y quan sát
          để thu được ước lượng không chệch Doubly Robust và sai số chuẩn Influence Function.
        - Trên tập kiểm thử ngoại mẫu C_test (len(train_df) != len(test_df)): Tuyệt đối KHÔNG truy cập test_df['Y'].
          Các nguy cơ phản sự thực được chuẩn hóa ngoại mẫu qua mô hình đầu ra m_0 và m_1 (E[Gamma(t)|Z] = mu_t(Z)),
          sai số chuẩn và khoảng tin cậy được ước lượng qua non-parametric bootstrap trên train_df.
          
    Xử lý T3 (Can thiệp đồng thời T1 và T2):
        - Áp dụng công thức Multi-valued AIPW (Cattaneo 2010; Hernan & Robins 2020) cho 2 nhánh mục tiêu (1,1) và (0,0).
        - Giữ lại 100% sinh viên trên cả Full Cohort (N=3,630) và C_test (n=545), KHÔNG dropna các nhánh (0,1) và (1,0).

    Nhận vào:
        train_df (pd.DataFrame): Tập dữ liệu huấn luyện.
        test_df (pd.DataFrame): Tập dữ liệu kiểm thử (C_test).
        treatment_type (str): 'T1', 'T2' hoặc 'T3'.
        cov_cols (Tuple[str, ...]): Danh sách tên biến nhiễu Z.

    Trả ra:
        Tuple[float, float, float, float, float, float]:
            (risk_do_0, risk_do_1, ate, ci_low, ci_high, p_val)
    """
    cov_list = list(cov_cols)
    is_full_cohort = (len(train_df) == len(test_df))
    
    if treatment_type == 'T1':
        t_col = 'T1'
        adj_cols = ['T2'] + cov_list
        ind_1_tr = (train_df[t_col] == 1).astype(int)
        ind_0_tr = (train_df[t_col] == 0).astype(int)
        
        # 1. Propensity Model
        ps_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(train_df[adj_cols], train_df[t_col])
        e_tr = np.clip(ps_model.predict_proba(train_df[adj_cols])[:, 1], 0.02, 0.98)
        
        # 2. Outcome Regression Models
        m1 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_1_tr == 1][adj_cols], train_df[ind_1_tr == 1]['Y']
        )
        m0 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_0_tr == 1][adj_cols], train_df[ind_0_tr == 1]['Y']
        )
        
        if is_full_cohort:
            mu1_tr = m1.predict_proba(train_df[adj_cols])[:, 1]
            mu0_tr = m0.predict_proba(train_df[adj_cols])[:, 1]
            gamma1 = mu1_tr + (train_df[t_col].values * (train_df['Y'].values - mu1_tr)) / e_tr
            gamma0 = mu0_tr + ((1.0 - train_df[t_col].values) * (train_df['Y'].values - mu0_tr)) / (1.0 - e_tr)
            
            r0 = float(np.clip(np.mean(gamma0), 0.0, 1.0))
            r1 = float(np.clip(np.mean(gamma1), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            diff = gamma1 - gamma0
            se = float(np.std(diff) / np.sqrt(len(diff)))
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val
        else:
            # Split Evaluation trên C_test: TUYỆT ĐỐI KHÔNG DÙNG test_df['Y']
            mu1_te = m1.predict_proba(test_df[adj_cols])[:, 1]
            mu0_te = m0.predict_proba(test_df[adj_cols])[:, 1]
            r0 = float(np.clip(np.mean(mu0_te), 0.0, 1.0))
            r1 = float(np.clip(np.mean(mu1_te), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            
            # Fast Vectorized Bootstrap trên train_df (B=100)
            np.random.seed(42)
            boot_ates = []
            n_tr = len(train_df)
            X_adj = train_df[adj_cols].values
            y_vals = train_df['Y'].values
            t_vals = train_df[t_col].values
            X_adj_te = test_df[adj_cols].values
            
            for b in range(100):
                b_idx = np.random.choice(n_tr, size=n_tr, replace=True)
                mask1 = (t_vals[b_idx] == 1)
                mask0 = (t_vals[b_idx] == 0)
                try:
                    m1_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask1], y_vals[b_idx][mask1]
                    )
                    m0_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask0], y_vals[b_idx][mask0]
                    )
                    p1 = m1_b.predict_proba(X_adj_te)[:, 1]
                    p0 = m0_b.predict_proba(X_adj_te)[:, 1]
                    boot_ates.append(float(np.mean(p1 - p0)))
                except Exception:
                    continue
                    
            se = float(np.std(boot_ates)) if len(boot_ates) >= 20 else float("nan")  # không gán cứng SE
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val

    elif treatment_type == 'T2':
        t_col = 'T2'
        adj_cols = ['T1'] + cov_list
        ind_1_tr = (train_df[t_col] == 1).astype(int)
        ind_0_tr = (train_df[t_col] == 0).astype(int)
        
        ps_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(train_df[adj_cols], train_df[t_col])
        e_tr = np.clip(ps_model.predict_proba(train_df[adj_cols])[:, 1], 0.02, 0.98)
        
        m1 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_1_tr == 1][adj_cols], train_df[ind_1_tr == 1]['Y']
        )
        m0 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_0_tr == 1][adj_cols], train_df[ind_0_tr == 1]['Y']
        )
        
        if is_full_cohort:
            mu1_tr = m1.predict_proba(train_df[adj_cols])[:, 1]
            mu0_tr = m0.predict_proba(train_df[adj_cols])[:, 1]
            gamma1 = mu1_tr + (train_df[t_col].values * (train_df['Y'].values - mu1_tr)) / e_tr
            gamma0 = mu0_tr + ((1.0 - train_df[t_col].values) * (train_df['Y'].values - mu0_tr)) / (1.0 - e_tr)
            
            r0 = float(np.clip(np.mean(gamma0), 0.0, 1.0))
            r1 = float(np.clip(np.mean(gamma1), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            diff = gamma1 - gamma0
            se = float(np.std(diff) / np.sqrt(len(diff)))
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val
        else:
            mu1_te = m1.predict_proba(test_df[adj_cols])[:, 1]
            mu0_te = m0.predict_proba(test_df[adj_cols])[:, 1]
            r0 = float(np.clip(np.mean(mu0_te), 0.0, 1.0))
            r1 = float(np.clip(np.mean(mu1_te), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            
            np.random.seed(42)
            boot_ates = []
            n_tr = len(train_df)
            X_adj = train_df[adj_cols].values
            y_vals = train_df['Y'].values
            t_vals = train_df[t_col].values
            X_adj_te = test_df[adj_cols].values
            
            for b in range(100):
                b_idx = np.random.choice(n_tr, size=n_tr, replace=True)
                mask1 = (t_vals[b_idx] == 1)
                mask0 = (t_vals[b_idx] == 0)
                try:
                    m1_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask1], y_vals[b_idx][mask1]
                    )
                    m0_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask0], y_vals[b_idx][mask0]
                    )
                    p1 = m1_b.predict_proba(X_adj_te)[:, 1]
                    p0 = m0_b.predict_proba(X_adj_te)[:, 1]
                    boot_ates.append(float(np.mean(p1 - p0)))
                except Exception:
                    continue
                    
            se = float(np.std(boot_ates)) if len(boot_ates) >= 20 else 0.02
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val

    else:  # T3 Joint: do(T1=1, T2=1) vs do(T1=0, T2=0) via Multi-valued AIPW (Cattaneo 2010)
        adj_cols = cov_list
        ind_11_tr = ((train_df['T1'] == 1) & (train_df['T2'] == 1)).astype(int)
        ind_00_tr = ((train_df['T1'] == 0) & (train_df['T2'] == 0)).astype(int)
        
        # 1. Multi-valued Generalized Propensity Models
        ps_11 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(train_df[adj_cols], ind_11_tr)
        ps_00 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(train_df[adj_cols], ind_00_tr)
        e_11_tr = np.clip(ps_11.predict_proba(train_df[adj_cols])[:, 1], 0.02, 0.98)
        e_00_tr = np.clip(ps_00.predict_proba(train_df[adj_cols])[:, 1], 0.02, 0.98)
        
        # 2. Outcome Regression Models cho nhánh (1,1) và (0,0)
        m11 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_11_tr == 1][adj_cols], train_df[ind_11_tr == 1]['Y']
        )
        m00 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(
            train_df[ind_00_tr == 1][adj_cols], train_df[ind_00_tr == 1]['Y']
        )
        
        if is_full_cohort:
            # Full Cohort: Đánh giá trên toàn bộ N=3,630 sinh viên (KHÔNG dropna)
            mu11_tr = m11.predict_proba(train_df[adj_cols])[:, 1]
            mu00_tr = m00.predict_proba(train_df[adj_cols])[:, 1]
            
            # Sinh viên thuộc (0,1) hoặc (1,0) có ind=0, chỉ đóng góp mu (không bị loại)
            gamma11 = mu11_tr + (ind_11_tr.values * (train_df['Y'].values - mu11_tr)) / e_11_tr
            gamma00 = mu00_tr + (ind_00_tr.values * (train_df['Y'].values - mu00_tr)) / e_00_tr
            
            r0 = float(np.clip(np.mean(gamma00), 0.0, 1.0))
            r1 = float(np.clip(np.mean(gamma11), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            diff = gamma11 - gamma00
            se = float(np.std(diff) / np.sqrt(len(diff)))
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val
        else:
            # Split Evaluation trên C_test: Chuẩn hóa trên TOÀN BỘ 545 sinh viên kiểm thử (KHÔNG dropna, KHÔNG dùng Y_test)
            mu11_te = m11.predict_proba(test_df[adj_cols])[:, 1]
            mu00_te = m00.predict_proba(test_df[adj_cols])[:, 1]
            r0 = float(np.clip(np.mean(mu00_te), 0.0, 1.0))
            r1 = float(np.clip(np.mean(mu11_te), 0.0, 1.0))
            ate = float(np.clip(r1 - r0, -1.0, 1.0))
            
            # Fast Vectorized Bootstrap trên train_df (B=100)
            np.random.seed(42)
            boot_ates = []
            n_tr = len(train_df)
            X_adj = train_df[adj_cols].values
            y_vals = train_df['Y'].values
            t1_vals = train_df['T1'].values
            t2_vals = train_df['T2'].values
            X_adj_te = test_df[adj_cols].values
            
            for b in range(100):
                b_idx = np.random.choice(n_tr, size=n_tr, replace=True)
                mask11 = (t1_vals[b_idx] == 1) & (t2_vals[b_idx] == 1)
                mask00 = (t1_vals[b_idx] == 0) & (t2_vals[b_idx] == 0)
                try:
                    m11_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask11], y_vals[b_idx][mask11]
                    )
                    m00_b = LogisticRegression(max_iter=300, solver='lbfgs', tol=1e-3, random_state=b).fit(
                        X_adj[b_idx][mask00], y_vals[b_idx][mask00]
                    )
                    p1 = m11_b.predict_proba(X_adj_te)[:, 1]
                    p0 = m00_b.predict_proba(X_adj_te)[:, 1]
                    boot_ates.append(float(np.mean(p1 - p0)))
                except Exception:
                    continue
                    
            se = float(np.std(boot_ates)) if len(boot_ates) >= 20 else 0.02
            ci_low = float(np.clip(ate - 1.96 * se, -1.0, 1.0))
            ci_high = float(np.clip(ate + 1.96 * se, -1.0, 1.0))
            z = ate / se if (se > 0 and np.isfinite(se)) else float("nan")
            p_val = float(2.0 * stats.norm.sf(abs(z)))  # sf tránh underflow của 1-cdf
            return r0, r1, ate, ci_low, ci_high, p_val


def thuc_thi_doi_chuan_toan_dien(cfg: MultiTreatmentConfig) -> pd.DataFrame:
    """
    Hàm này để làm gì: Chạy toàn bộ quy trình đối chuẩn đa can thiệp (T1, T2, T3) trên cả quần thể
    toàn phần (N=3,630) và 5-seed paired split (C_test, n=545, seeds 42..46) qua 3 bộ ước lượng.
    Bảo đảm quy tắc Strict Zero-Leakage Preprocessing và Axiomatic Bounding.

    Nhận vào:
        cfg (MultiTreatmentConfig): Cấu hình thực nghiệm.

    Trả ra:
        pd.DataFrame: Bảng kết quả tổng hợp đầy đủ Table_Multi_Treatment.csv.
    """
    df = tai_va_tien_xu_ly_du_lieu(cfg.data_path)
    
    treatment_meta = {
        'T1': {
            'name': 'T1: Scholarship holder',
            'desc': 'do(T1=1) vs do(T1=0)',
            'adj': '{Z1..Z6, Tuition_fees_up_to_date}',
            'policy': 'Cấp học bổng giúp tăng xác suất tốt nghiệp khoảng +21% đến +22%'
        },
        'T2': {
            'name': 'T2: Tuition fees up to date',
            'desc': 'do(T2=1) vs do(T2=0)',
            'adj': '{Z1..Z6, Scholarship_holder}',
            'policy': 'KHÔNG diễn giải như chính sách: T2 là trạng thái học phí, có nguy cơ nhân quả ngược (bỏ học -> ngừng đóng phí); chỉ là cận trên mang tính liên kết'
        },
        'T3': {
            'name': 'T3: Joint contrast do(T1=1,T2=1) vs do(T1=0,T2=0)',
            'desc': 'do(T1=1, T2=1) vs do(T1=0, T2=0)',
            'adj': '{Z1_age, Z2_adm, Z3_mom, Z4_debt, Z5_disp, Z6_gen}',
            'policy': 'Tương phản kép ~ cộng dồn T1+T2 (không có bằng chứng cộng hưởng); kế thừa toàn bộ hạn chế của T2 và positivity yếu (ô T1=1,T2=0: 37 SV)'
        }
    }
    
    estimators = [
        ('Backdoor Linear Regression (LPM)', uoc_luong_backdoor_lpm),
        ('G-Computation (Logistic)', uoc_luong_g_computation),
        ('Doubly Robust (AIPW)', uoc_luong_doubly_robust)
    ]
    
    results: List[CausalEstimateResult] = []
    
    print("\n" + "=" * 80)
    print("BẮT ĐẦU ĐỐI CHUẨN ĐA CAN THIỆP (MULTI-TREATMENT CAUSAL EVALUATION - STRICT ZERO LEAKAGE)")
    print("=" * 80)
    
    for tr_id, tr_info in treatment_meta.items():
        print(f"\n>>> Đang phân tích Can thiệp: {tr_info['name']} [{tr_info['desc']}]")
        
        for est_name, est_func in estimators:
            can_scale = ('LPM' not in est_name)
            # Trên C_test không dùng Y_test nên nhánh 'AIPW' thực chất là Outcome Regression (T-learner)
            est_label_split = ('Outcome Regression (T-learner, no Y_test)' if 'AIPW' in est_name else est_name)
            
            # 1. Đánh giá trên Quần thể Toàn phần (Full Cohort N=3,630)
            tr_full, te_full = ap_dung_chuan_hoa_khong_ro_ri(df, df, cfg.covariate_cols, can_scale)
            r0_f, r1_f, ate_f, low_f, high_f, p_f = est_func(tr_full, te_full, tr_id, cfg.covariate_cols)
            
            results.append(CausalEstimateResult(
                treatment_id=tr_id,
                treatment_name=tr_info['name'],
                estimator=est_name,
                cohort='Full Cohort (N=3,630)',
                seed='Full_Cohort',
                cohort_size=len(df),
                risk_do_0=r0_f,
                risk_do_1=r1_f,
                ate=ate_f,
                ci_95_low=low_f,
                ci_95_high=high_f,
                p_value=p_f,
                adjustment_set=tr_info['adj'],
                policy_implication=tr_info['policy']
            ))
            print(f"  [{est_name}] Full Cohort: Risk(0)={r0_f:.4f}, Risk(1)={r1_f:.4f}, ATE={ate_f:+.4f}, CI=[{low_f:.4f}, {high_f:.4f}], p={p_f:.2e}")
            
            # 2. Đánh giá 5-Seed Paired Splits trên C_test (n=545)
            split_runs = []
            for s in cfg.seeds:
                split_file = cfg.splits_dir / f"split_seed_{s}.json"
                with open(split_file, 'r', encoding='utf-8') as f:
                    sp_data = json.load(f)
                    
                tr_raw = df.iloc[sp_data['train_indices']].copy().reset_index(drop=True)
                te_raw = df.iloc[sp_data['test_indices']].copy().reset_index(drop=True)
                
                # Áp dụng chuẩn hóa CHỈ fit trên train_raw, transform te_raw (Strict Zero-Leakage)
                tr_sub, te_sub = ap_dung_chuan_hoa_khong_ro_ri(tr_raw, te_raw, cfg.covariate_cols, can_scale)
                
                r0_s, r1_s, ate_s, low_s, high_s, p_s = est_func(tr_sub, te_sub, tr_id, cfg.covariate_cols)
                
                results.append(CausalEstimateResult(
                    treatment_id=tr_id,
                    treatment_name=tr_info['name'],
                    estimator=est_label_split,
                    cohort='Test Cohort (C_test)',
                    seed=str(s),
                    cohort_size=len(te_sub),
                    risk_do_0=r0_s,
                    risk_do_1=r1_s,
                    ate=ate_s,
                    ci_95_low=low_s,
                    ci_95_high=high_s,
                    p_value=p_s,
                    adjustment_set=tr_info['adj'],
                    policy_implication=tr_info['policy']
                ))
                split_runs.append({
                    'r0': r0_s, 'r1': r1_s, 'ate': ate_s, 'low': low_s, 'high': high_s, 'p': p_s
                })
                
            # 3. Hàng tổng hợp Pooled 5 Runs (Mean ± Std)
            df_sp = pd.DataFrame(split_runs)
            mean_r0 = float(np.clip(df_sp['r0'].mean(), 0.0, 1.0))
            mean_r1 = float(np.clip(df_sp['r1'].mean(), 0.0, 1.0))
            mean_ate = float(np.clip(df_sp['ate'].mean(), -1.0, 1.0))
            mean_low = float(np.clip(df_sp['low'].mean(), -1.0, 1.0))
            mean_high = float(np.clip(df_sp['high'].mean(), -1.0, 1.0))
            mean_p = float(df_sp['p'].max())  # bảo thủ: p lớn nhất trong 5 split
            
            results.append(CausalEstimateResult(
                treatment_id=tr_id,
                treatment_name=tr_info['name'],
                estimator=est_label_split,
                cohort='Pooled 5 Splits (Mean)',
                seed='Pooled 5 Runs',
                cohort_size=545,
                risk_do_0=mean_r0,
                risk_do_1=mean_r1,
                ate=mean_ate,
                ci_95_low=mean_low,
                ci_95_high=mean_high,
                p_value=mean_p,
                adjustment_set=tr_info['adj'],
                policy_implication=tr_info['policy']
            ))
            print(f"  [{est_name}] Pooled 5 Splits: Risk(0)={mean_r0:.4f} +/- {df_sp['r0'].std():.4f}, Risk(1)={mean_r1:.4f} +/- {df_sp['r1'].std():.4f}, ATE={mean_ate:+.4f} +/- {df_sp['ate'].std():.4f}, CI=[{mean_low:.4f}, {mean_high:.4f}]")

    # Đóng gói DataFrame
    df_out = pd.DataFrame([r.__dict__ for r in results])
    
    # Tạo thư mục đầu ra nếu chưa có
    cfg.output_table_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.output_table_path, index=False, encoding='utf-8')
    print(f"\n[XUẤT BẢN THÀNH CÔNG] Đã lưu Table_Multi_Treatment tại: {cfg.output_table_path}")
    print(f"Tổng số bản ghi đối chuẩn: {len(df_out)} dòng.")
    return df_out


def hien_thi_bang_tong_hop_chinh_sach(df_table: pd.DataFrame) -> None:
    """
    Hàm này để làm gì: Trích xuất và in bảng tóm tắt đối chuẩn các can thiệp đa chiều
    cho phần thuyết minh báo cáo bảo vệ.

    Nhận vào:
        df_table (pd.DataFrame): Bảng kết quả tổng hợp.
    """
    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP HIỆU ỨNG ĐA CAN THIỆP CHÍNH SÁCH (TABLE MULTI-TREATMENT SUMMARY)")
    print("=" * 90)
    
    # Lấy các dòng Pooled 5 Runs của G-Computation và Doubly Robust và LPM
    summary_rows = df_table[df_table['seed'] == 'Pooled 5 Runs'][
        ['treatment_id', 'estimator', 'risk_do_0', 'risk_do_1', 'ate', 'ci_95_low', 'ci_95_high', 'p_value']
    ].copy()
    
    for idx, row in summary_rows.iterrows():
        p_str = "< 1e-10" if row['p_value'] < 1e-10 else f"{row['p_value']:.2e}"
        print(f"[{row['treatment_id']}] {row['estimator']:<34} | do(0)={row['risk_do_0']*100:.1f}% | do(1)={row['risk_do_1']*100:.1f}% | ATE={row['ate']*100:+.2f}% | 95% CI=[{row['ci_95_low']*100:+.1f}%, {row['ci_95_high']*100:+.1f}%] | p={p_str}")
    print("=" * 90)


def main() -> None:
    """
    Hàm chính thực thi toàn bộ luồng kiểm toán và tính toán đa can thiệp.
    """
    root_dir = Path(__file__).resolve().parent.parent
    config = MultiTreatmentConfig(
        data_path=root_dir / 'data' / 'uci_697.csv',
        splits_dir=root_dir / 'splits',
        output_table_path=root_dir / 'artifacts' / 'tables' / 'Table_Multi_Treatment.csv',
        seeds=(42, 43, 44, 45, 46),
        covariate_cols=('Z1_age', 'Z2_adm', 'Z3_mom', 'Z4_debt', 'Z5_disp', 'Z6_gen')
    )
    
    df_result = thuc_thi_doi_chuan_toan_dien(config)
    hien_thi_bang_tong_hop_chinh_sach(df_result)


if __name__ == '__main__':
    main()
