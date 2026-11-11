# -*- coding: utf-8 -*-
"""
NHÓM 13 — Mô phỏng phân bổ học bổng theo CATE (mục 10)

Mã hóa biến nền và so sánh 4 cách phân bổ cùng số suất học bổng bằng T-learner logistic (in-sample).

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - ma_hoa_z_chinh_sach                        Cell 86 (def)
  - danh_gia_chinh_sach                        Cell 86 (def)
"""


# ==============================================================================================================
# ma_hoa_z_chinh_sach  (def)
# Vị trí: Cell 86 · mục 10.2 Kiểm chứng chuyển dịch ngoại miền OOD trên Kaggle S4E6 ($N = 76,518$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Mã hóa 5 biến nền cho mô phỏng phân bổ (cùng mã hóa với mục 9.H), kèm T và Y.
# ==============================================================================================================
def ma_hoa_z_chinh_sach(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Mã hóa 5 biến nền dùng cho mô phỏng phân bổ học bổng (cùng mã hóa với mục 9.H).

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        pd.DataFrame: Cột Z1..Z5, T, Y.
    """
    d = du_lieu.reset_index(drop=True)
    return pd.DataFrame({
        'Z1': pd.qcut(d['Age_at_enrollment'], q=3, labels=[0, 1, 2]).astype(int),
        'Z2': (d['Admission_grade'] >= d['Admission_grade'].median()).astype(int),
        'Z3': (d['Mothers_qualification'] > 1).astype(int),
        'Z4': d['Debtor'].astype(int),
        'Z5': d['Tuition_fees_up_to_date'].astype(int),
        'T': d['Scholarship_holder'].astype(int),
        'Y': d['Target_Graduate'].astype(int),
    })


# ==============================================================================================================
# danh_gia_chinh_sach  (def)
# Vị trí: Cell 86 · mục 10.2 Kiểm chứng chuyển dịch ngoại miền OOD trên Kaggle S4E6 ($N = 76,518$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : So sánh 4 cách phân bổ CÙNG số suất học bổng (969): theo CATE, thực tế, ưu tiên Debtor, ngẫu nhiên;
#            ước lượng tỷ lệ tốt nghiệp dự kiến.
# CÁCH LÀM :
#   1. T-learner: logistic riêng cho T=0 (mu0) và T=1 (mu1) trên toàn bộ dữ liệu.
#   2. CATE ước lượng = mu1 - mu0 cho từng SV.
#   3. Mỗi chính sách là vector pi (1 = được cấp): CATE optimal lấy 969 người có CATE cao nhất; Need-based xếp
#      theo Debtor rồi học phí.
#   4. Giá trị chính sách V(pi) = trung bình[pi x mu1 + (1 - pi) x mu0]; so số SV tốt nghiệp với phân bổ thực
#      tế.
# LƯU Ý    : In-sample: cùng mu0, mu1 vừa dùng để chọn vừa dùng để chấm, chưa cross-fitting nên +164 SV có xu
#            hướng lạc quan. Xếp hạng còn dùng biến học phí (có thể là hậu quả của bỏ học).
# ❓ HỎI   : +164 SV có đáng tin không?
#   ➜ ĐÁP : Là ước lượng mô hình, không phải quan sát; cần cross-fitting và đánh giá off-policy (IPW/DR) mới
#           chắc.
# ==============================================================================================================
def danh_gia_chinh_sach(df_base: pd.DataFrame, so_suat: int, seed: int = 42) -> pd.DataFrame:
    """
    Hàm này để làm gì: Ước lượng tỷ lệ tốt nghiệp dự kiến V(pi) = mean[pi*mu1 + (1-pi)*mu0] của 4 cách phân bổ
    cùng số suất học bổng: theo CATE, thực tế, ưu tiên nợ học phí, ngẫu nhiên.
    Lưu ý: mu0, mu1 khớp trên TOÀN BỘ dữ liệu và dùng lại để chấm điểm (in-sample, chưa cross-fit).

    Nhận vào:
        df_base (pd.DataFrame): Kết quả của ma_hoa_z_chinh_sach.
        so_suat (int): Số suất học bổng (ngân sách).
        seed (int): Hạt giống cho chính sách ngẫu nhiên.

    Trả ra:
        pd.DataFrame: Một dòng mỗi chính sách.
    """
    cov = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    m0 = LogisticRegression(max_iter=500).fit(df_base[df_base['T'] == 0][cov], df_base[df_base['T'] == 0]['Y'])
    m1 = LogisticRegression(max_iter=500).fit(df_base[df_base['T'] == 1][cov], df_base[df_base['T'] == 1]['Y'])
    mu0 = m0.predict_proba(df_base[cov])[:, 1]
    mu1 = m1.predict_proba(df_base[cov])[:, 1]
    cate = mu1 - mu0
    n = len(df_base)
    pol = {
        'Chính sách Tối ưu Nhân quả (CATE Optimal)': np.isin(np.arange(n), np.argsort(-cate)[:so_suat]).astype(float),
        'Chính sách Thực tế (Factual Allocation)': df_base['T'].values.astype(float),
        'Chính sách Ưu tiên Cần thiết (Need-based Debtor)': np.isin(
            np.arange(n), df_base.sort_values(['Z4', 'Z5'], ascending=[False, True]).index[:so_suat]).astype(float),
    }
    rng = np.random.RandomState(seed)
    pol['Chính sách Phân bổ Ngẫu nhiên (Random)'] = np.isin(np.arange(n), rng.choice(n, size=so_suat, replace=False)).astype(float)
    gia_tri = {k: mu1 * p + mu0 * (1 - p) for k, p in pol.items()}
    goc = gia_tri['Chính sách Thực tế (Factual Allocation)'].sum()
    dong = []
    for k, v in gia_tri.items():
        dong.append({'Chính sách Phân bổ': k, 'Tỷ lệ Cấp': f'{so_suat / n * 100:.1f}% ({so_suat} SV)',
                     'Tỷ lệ Tốt nghiệp Dự kiến': f'{v.mean() * 100:.2f}%', 'Số SV Tốt nghiệp': int(v.sum()),
                     'Số SV Tăng thêm vs Factual': 'Baseline' if 'Factual' in k else f'{int(v.sum()) - int(goc):+d} SV'})
    return pd.DataFrame(dong)
