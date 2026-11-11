# -*- coding: utf-8 -*-
"""
NHÓM 01 — Nạp dữ liệu UCI 697 và kiểm toán trước khi mô hình hóa (mục 1)

Đọc dữ liệu gốc, lọc quần thể Graduate/Dropout, ghi môi trường chạy, báo cáo luồng loại trừ, phân loại biến
theo thời điểm đo, vẽ EDA (Hình 1.1, 1.2) và kiểm tra overlap thô (Bảng 1.6b).

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - _dinh_dang_so                              Cell  4 (def)
  - ThongKeTapDuLieu                           Cell  4 (class)
  - doc_va_kiem_toan_du_lieu_uci               Cell  4 (def)
  - in_thong_tin_moi_truong                    Cell  6 (def)
  - luong_loai_tru                             Cell  9 (def)
  - kiem_toan_thoi_diem_bien                   Cell 11 (def)
  - ve_phan_bo_diem_vao                        Cell 12 (def)
  - ve_mutual_information                      Cell 13 (def)
  - bang_overlap_theo_tang                     Cell 14 (def)
"""


# ==============================================================================================================
# _dinh_dang_so  (def)
# Vị trí: Cell 4 · mục 1.2–1.4 Metadata, nạp và kiểm toán dữ liệu UCI 697
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Định dạng số khi HIỂN THỊ bảng pandas: 4 chữ số thập phân, số nguyên in gọn, số rất nhỏ (p-value)
#            dạng khoa học.
# CÁCH LÀM :
#   1. NaN thì in "NaN" (x != x chỉ đúng với NaN).
#   2. Số nguyên (ví dụ 4424.0) in "4424".
#   3. Số khác 0 nhưng nhỏ hơn 1e-4 (p-value 4.4e-41) in dạng khoa học để không bị thành 0.0000.
#   4. Còn lại in 4 chữ số thập phân.
# LƯU Ý    : Được đăng ký bằng pd.set_option("display.float_format", ...). Chỉ đổi cách hiển thị; file CSV vẫn
#            đủ độ chính xác.
# ==============================================================================================================
def _dinh_dang_so(x: float) -> str:
    """Hiển thị số thực trong bảng: 4 chữ số thập phân; số nguyên in gọn; số rất nhỏ (p-value) giữ dạng khoa học."""
    if x != x:  # NaN
        return 'NaN'
    if float(x).is_integer() and abs(x) < 1e15:
        return f'{x:.0f}'
    if x != 0 and abs(x) < 1e-4:
        return f'{x:.2e}'
    return f'{x:.4f}'


# ==============================================================================================================
# ThongKeTapDuLieu  (class)
# Vị trí: Cell 4 · mục 1.2–1.4 Metadata, nạp và kiểm toán dữ liệu UCI 697
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass gom các con số tóm tắt quần thể: số mẫu, số tốt nghiệp/bỏ học, số nhận học bổng và tỷ lệ.
# LƯU Ý    : Chỉ là "hộp chứa" dữ liệu, không có logic.
# ==============================================================================================================
@dataclass
class ThongKeTapDuLieu:
    tong_so_mau: int
    so_thuoc_tinh: int
    so_tot_nghiep: int
    so_bo_hoc: int
    ty_le_tot_nghiep: float
    so_nhan_hoc_bong: int
    ty_le_hoc_bong: float


# ==============================================================================================================
# doc_va_kiem_toan_du_lieu_uci  (def)
# Vị trí: Cell 4 · mục 1.2–1.4 Metadata, nạp và kiểm toán dữ liệu UCI 697
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Đọc file UCI 697, chuẩn hóa tên cột, lọc quần thể Graduate/Dropout và tạo nhãn nhị phân
#            Target_Graduate.
# ĐẦU VÀO  : duong_dan_tep: đường dẫn data/uci_697/data.csv.
# ĐẦU RA   : (df_goc 4,424 dòng, df_nhi_phan 3,630 dòng, thong_ke).
# CÁCH LÀM :
#   1. Đọc CSV.
#   2. Chuẩn hóa tên cột: khoảng trắng -> "_", bỏ dấu nháy, ngoặc; "/" -> "_" (ví dụ "Mother's qualification" ->
#      "Mothers_qualification").
#   3. Giữ dòng có Target là Dropout hoặc Graduate, tức là loại 794 Enrolled (right-censoring: chưa có kết cục).
#   4. Tạo Target_Graduate = 1 nếu Graduate, 0 nếu Dropout.
#   5. Đếm số tốt nghiệp, bỏ học, nhận học bổng và đóng gói vào ThongKeTapDuLieu.
# ❓ HỎI   : Vì sao loại Enrolled?
#   ➜ ĐÁP : Họ chưa có kết cục lúc chốt dữ liệu; gán 0 hay 1 đều là đoán. Mục 9.D kiểm tra lại trên cả 4,424 SV
#           bằng kết cục giữ chân: +16.87%.
# ==============================================================================================================
def doc_va_kiem_toan_du_lieu_uci(duong_dan_tep: Path) -> Tuple[pd.DataFrame, pd.DataFrame, ThongKeTapDuLieu]:
    """
    Hàm này để làm gì: Đọc dữ liệu thực tế UCI ID 697 từ tệp CSV, chuẩn hóa tên cột
    và tạo tập dữ liệu nhị phân (Graduate vs Dropout) cho bài toán phân tích nhân quả.

    Nhận vào:
        duong_dan_tep (Path): Đường dẫn tệp tin CSV dữ liệu UCI 697.

    Trả ra:
        Tuple[pd.DataFrame, pd.DataFrame, ThongKeTapDuLieu]:
            - df_goc: Bảng dữ liệu toàn bộ sinh viên (N=4,424).
            - df_nhi_phan: Bảng dữ liệu sinh viên có kết quả dứt khoát (N=3,630).
            - thong_ke: Đối tượng dataclass chứa các chỉ số tóm tắt quần thể.
    """
    df_goc = pd.read_csv(duong_dan_tep, sep=',')
    
    # Chuẩn hóa tên cột: thay khoảng trắng, dấu ngoặc đơn và dấu gạch chéo bằng dấu gạch dưới
    df_goc.columns = [
        c.strip().replace(' ', '_').replace("'", "").replace('(', '').replace(')', '').replace('/', '_')
        for c in df_goc.columns
    ]
    
    # Lọc quần thể nhị phân: loại bỏ sinh viên còn đang theo học (Enrolled) để xác định rõ kết quả
    df_nhi_phan = df_goc[df_goc['Target'].isin(['Dropout', 'Graduate'])].copy()
    df_nhi_phan['Target_Graduate'] = (df_nhi_phan['Target'] == 'Graduate').astype(int)
    
    n_total = len(df_nhi_phan)
    n_grad = int(df_nhi_phan['Target_Graduate'].sum())
    n_drop = n_total - n_grad
    n_schol = int(df_nhi_phan['Scholarship_holder'].sum())
    
    thong_ke = ThongKeTapDuLieu(
        tong_so_mau=n_total,
        so_thuoc_tinh=df_nhi_phan.shape[1],
        so_tot_nghiep=n_grad,
        so_bo_hoc=n_drop,
        ty_le_tot_nghiep=float(n_grad / n_total),
        so_nhan_hoc_bong=n_schol,
        ty_le_hoc_bong=float(n_schol / n_total)
    )
    
    return df_goc, df_nhi_phan, thong_ke


# ==============================================================================================================
# in_thong_tin_moi_truong  (def)
# Vị trí: Cell 6 · mục 1.0b Cài đặt, tái lập và thời gian chạy
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Ghi lại phiên bản Python, thư viện, CPU và revision git của lần chạy (yêu cầu tái lập, Slide 41,
#            76).
# CÁCH LÀM :
#   1. Lấy phiên bản Python, hệ điều hành, CPU.
#   2. Import từng gói và đọc __version__; gói nào thiếu thì ghi "không có".
#   3. Gọi git rev-parse để lấy mã commit; nếu có thay đổi chưa commit thì ghi thêm chú thích.
# LƯU Ý    : Giúp người khác dựng lại đúng môi trường; không ảnh hưởng kết quả.
# ==============================================================================================================
def in_thong_tin_moi_truong(cac_goi: List[str]) -> pd.DataFrame:
    """
    Hàm này để làm gì: Ghi lại phiên bản Python, thư viện, revision git và phần cứng của lần chạy (Slide 41, 76).

    Nhận vào:
        cac_goi (List[str]): Tên các gói cần ghi phiên bản.

    Trả ra:
        pd.DataFrame: Bảng (mục, giá trị).
    """
    dong: List[Tuple[str, str]] = [('python', platform.python_version()),
                                   ('platform', platform.platform()),
                                   ('cpu', f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical)")]
    for goi in cac_goi:
        try:
            dong.append((goi, importlib.import_module(goi).__version__))
        except Exception as loi:
            dong.append((goi, f'không có ({type(loi).__name__})'))
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
        ban = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=10).stdout.strip()
        dong.append(('git revision', f"{rev}{' (+ thay đổi chưa commit)' if ban else ''}" or 'không xác định'))
    except Exception:
        dong.append(('git revision', 'không xác định'))
    return pd.DataFrame(dong, columns=['Mục', 'Giá trị'])


# ==============================================================================================================
# luong_loai_tru  (def)
# Vị trí: Cell 9 · mục 1.5 Câu hỏi nhân quả, từ điển biến và bảng giả định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng luồng loại trừ (eligibility flow, Slide 55): 4,424 -> loại 794 Enrolled -> 3,630, kèm kiểm tra
#            chất lượng.
# CÁCH LÀM :
#   1. Đếm số dòng gốc và sau khi lọc Dropout/Graduate.
#   2. Đếm dòng trùng lặp hoàn toàn (duplicated trên mọi cột) -> 0.
#   3. Đếm giá trị thiếu ở 7 cột dùng trong mô hình -> 0.
#   4. Kiểm tra điểm đầu vào nằm trong [95, 200] -> 0 dòng ngoài miền.
#   5. Tính tỷ lệ tốt nghiệp 60.85%.
# ❓ HỎI   : Có xử lý missing/outlier không?
#   ➜ ĐÁP : Không cần: 0 dòng trùng, 0 giá trị thiếu, 0 điểm ngoài miền hợp lệ; chỉ loại Enrolled.
# ==============================================================================================================
def luong_loai_tru(du_lieu_goc: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Báo cáo số dòng trước/sau từng bước loại trừ, số dòng trùng lặp và giá trị thiếu (Slide 55).

    Nhận vào:
        du_lieu_goc (pd.DataFrame): Toàn bộ dữ liệu UCI 697 (4,424 dòng).

    Trả ra:
        pd.DataFrame: Bảng luồng loại trừ và kiểm tra chất lượng.
    """
    cot = ['Scholarship_holder', 'Target', 'Admission_grade', 'Debtor', 'Tuition_fees_up_to_date',
           'Mothers_qualification', 'Age_at_enrollment']
    sau_loc = du_lieu_goc[du_lieu_goc['Target'].isin(['Dropout', 'Graduate'])]
    return pd.DataFrame([
        ('Dữ liệu gốc UCI 697', len(du_lieu_goc)),
        ('Loại Target = Enrolled (chưa có kết cục)', len(du_lieu_goc) - len(sau_loc)),
        ('Quần thể phân tích', len(sau_loc)),
        ('Dòng trùng lặp hoàn toàn (toàn bộ cột)', int(du_lieu_goc.duplicated().sum())),
        ('Giá trị thiếu ở các biến dùng trong mô hình', int(sau_loc[cot].isna().sum().sum())),
        ('Admission grade ngoài [95, 200]', int((~sau_loc['Admission_grade'].between(95, 200)).sum())),
        ('Tỷ lệ tốt nghiệp (%)', round(100 * (sau_loc['Target'] == 'Graduate').mean(), 2)),
    ], columns=['Bước / kiểm tra', 'Giá trị'])


# ==============================================================================================================
# kiem_toan_thoi_diem_bien  (def)
# Vị trí: Cell 11 · mục 1.6 Kiểm toán dữ liệu bổ sung trước khi mô hình hóa
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng 1.6a: phân loại mọi cột theo thời điểm đo so với học bổng: biến nền (được làm Z) và biến sau
#            nhập học (cấm điều chỉnh).
# CÁCH LÀM :
#   1. Lấy mọi cột trừ Target.
#   2. Cột bắt đầu bằng "Curricular_units" (tín chỉ, điểm HK1/HK2) là biến SAU nhập học: 12 biến.
#   3. Các cột còn lại là biến nền: 24 biến.
#   4. Ghi rõ cột nào đang được dùng làm T/Z1..Z5 và quy tắc cho từng nhóm.
# LÝ THUYẾT: Điều chỉnh theo biến hậu can thiệp (post-treatment) sẽ chặn mất một phần hiệu ứng đi qua nó, làm
#            ATE bị chệch.
# ❓ HỎI   : Sao không dùng điểm HK1/HK2 làm biến điều chỉnh, dù chúng dự báo rất tốt?
#   ➜ ĐÁP : Chúng xảy ra sau học bổng, là trung gian T -> M -> Y. Điều chỉnh theo chúng sẽ "ăn mất" hiệu ứng.
# ==============================================================================================================
def kiem_toan_thoi_diem_bien(du_lieu_goc: pd.DataFrame, bien_z: Dict[str, str]) -> pd.DataFrame:
    """
    Hàm này để làm gì: Phân loại mọi cột của UCI 697 theo thời điểm đo so với can thiệp T (học bổng lúc nhập học):
    biến nền (đo trước/lúc nhập học, được phép làm Z) và biến sau nhập học (tín chỉ HK1/HK2, KHÔNG được điều chỉnh).

    Nhận vào:
        du_lieu_goc (pd.DataFrame): Toàn bộ 4,424 dòng UCI 697.
        bien_z (Dict[str, str]): Cột được chọn làm Z/T/Y trong mô hình -> vai trò.

    Trả ra:
        pd.DataFrame: Bảng tổng hợp theo nhóm thời điểm.
    """
    cot = [c.strip().lstrip('﻿') for c in du_lieu_goc.columns if c != 'Target']
    sau = [c for c in cot if c.startswith('Curricular_units')]
    nen = [c for c in cot if c not in sau]
    dong = [
        {'Nhóm': 'Biến nền (đo trước/lúc nhập học)', 'Số biến': len(nen),
         'Dùng trong mô hình': ', '.join(f'{c} ({bien_z[c]})' for c in nen if c in bien_z),
         'Quy tắc': 'Được phép làm Z; chỉ chọn biến có lý do ảnh hưởng cả T và Y'},
        {'Nhóm': 'Biến sau nhập học (tín chỉ, điểm HK1/HK2)', 'Số biến': len(sau),
         'Dùng trong mô hình': 'Curricular_units_1st_sem_approved (chỉ làm trung gian M1 ở mục 9)',
         'Quy tắc': 'KHÔNG đưa vào tập điều chỉnh — xảy ra sau T, điều chỉnh sẽ chặn mất hiệu ứng'},
    ]
    return pd.DataFrame(dong)


# ==============================================================================================================
# ve_phan_bo_diem_vao  (def)
# Vị trí: Cell 12 · mục 1.6 Kiểm toán dữ liệu bổ sung trước khi mô hình hóa
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 1.1: phân bố điểm đầu vào với hai ngưỡng 120/140 của Z1, và tỷ lệ tốt nghiệp trong từng nhóm.
# CÁCH LÀM :
#   1. Chia điểm vào thành 3 nhóm bằng pd.cut: <120, [120,140), >=140 (right=False nghĩa là khoảng đóng bên
#      trái).
#   2. Trái: histogram 30 cột, kẻ hai vạch đỏ tại 120 và 140.
#   3. Phải: cột tỷ lệ tốt nghiệp theo nhóm, ghi n trên mỗi cột.
#   4. Trả bảng số SV, tỷ lệ học bổng, tỷ lệ tốt nghiệp theo nhóm.
# ❓ HỎI   : Ngưỡng 120/140 chọn thế nào?
#   ➜ ĐÁP : Chốt trước theo ý nghĩa thực tế, không dò trên dữ liệu; hình kiểm tra lại là 3 nhóm đều đủ mẫu
#           (1,075 / 1,869 / 686).
# ==============================================================================================================
def ve_phan_bo_diem_vao(du_lieu: pd.DataFrame, duong_dan_luu: Path) -> pd.DataFrame:
    """
    Hàm này để làm gì: Vẽ phân bố điểm đầu vào (N=3,630) kèm hai ngưỡng rời rạc hóa 120 / 140 của Z1, và tỷ lệ tốt nghiệp
    trong từng nhóm điểm, để kiểm tra ngưỡng chia có đủ mẫu và có phân biệt kết cục không.

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.
        duong_dan_luu (Path): Đường dẫn lưu PNG.

    Trả ra:
        pd.DataFrame: Số sinh viên, tỷ lệ học bổng và tỷ lệ tốt nghiệp theo nhóm điểm.
    """
    nhom = pd.cut(du_lieu['Admission_grade'], bins=[-np.inf, 120, 140, np.inf], labels=['< 120', '120–140', '≥ 140'], right=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1.5, 1]})
    sns.histplot(du_lieu['Admission_grade'], bins=30, ax=ax1, color='#5a7bb5')
    for nguong in (120, 140):
        ax1.axvline(nguong, color='#c53030', ls='--', lw=1.4)
        ax1.text(nguong + 0.8, ax1.get_ylim()[1] * 0.92, f'ngưỡng {nguong}', color='#c53030', fontsize=9)
    ax1.set_xlabel('Điểm đầu vào (Admission grade)')
    ax1.set_ylabel('Số sinh viên')
    ax1.set_title('Phân bố điểm đầu vào và ngưỡng chia Z1')
    bang = du_lieu.groupby(nhom, observed=True).agg(So_SV=('Target_Graduate', 'size'),
                                                     Ty_le_hoc_bong=('Scholarship_holder', 'mean'),
                                                     Ty_le_tot_nghiep=('Target_Graduate', 'mean'))
    cot = ax2.bar(bang.index.astype(str), bang['Ty_le_tot_nghiep'] * 100, color='#2f855a', width=0.55)
    for c, (n, r) in zip(cot, bang[['So_SV', 'Ty_le_tot_nghiep']].values):
        ax2.text(c.get_x() + c.get_width() / 2, r * 100 + 1.5, f'{r*100:.1f}%\nn={int(n):,}', ha='center', fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Tỷ lệ tốt nghiệp (%)')
    ax2.set_title('Tỷ lệ tốt nghiệp theo nhóm Z1')
    fig.suptitle('Hình 1.1. Điểm đầu vào: phân bố và nhóm rời rạc hóa', fontsize=11)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()
    return bang.round(3)


# ==============================================================================================================
# ve_mutual_information  (def)
# Vị trí: Cell 13 · mục 1.6 Kiểm toán dữ liệu bổ sung trước khi mô hình hóa
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 1.2: xếp hạng biến nền theo Mutual Information với Y; tô xanh biến đang dùng, đỏ biến học phí,
#            xám biến không dùng.
# CÁCH LÀM :
#   1. Bỏ Target và mọi biến "Curricular_units" (sau nhập học).
#   2. Ép các cột về số, ô lỗi thay bằng 0.
#   3. mutual_info_classif ước lượng MI (k-NN, random_state=42).
#   4. Lấy 15 biến cao nhất và vẽ thanh ngang theo màu.
# LÝ THUYẾT: I(X;Y) = sum p(x,y) log[p(x,y) / (p(x)p(y))]; bằng 0 khi độc lập. MI đo LIÊN KẾT, không đo nhân
#            quả.
# LƯU Ý    : mutual_info_classif coi mọi cột là liên tục (ước lượng k-NN), kể cả cột định danh như Course, nên
#            chỉ dùng để xếp hạng tương đối.
# ❓ HỎI   : Có chọn Z theo MI không?
#   ➜ ĐÁP : Không. Z chọn theo tiêu chuẩn cửa sau (thời điểm + ảnh hưởng cả T và Y); MI chỉ để kiểm tra lại.
# ==============================================================================================================
def ve_mutual_information(du_lieu: pd.DataFrame, cac_bien_z: List[str], duong_dan_luu: Path, top: int = 15) -> pd.DataFrame:
    """
    Hàm này để làm gì: Ước lượng Mutual Information giữa từng biến NỀN (bỏ biến tín chỉ sau nhập học) và kết cục Y
    trên N=3,630; tô màu biến đang dùng làm Z và biến học phí. Đây là mức LIÊN KẾT, không phải hiệu ứng nhân quả.

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.
        cac_bien_z (List[str]): Các cột đang được dùng làm Z hoặc T.
        duong_dan_luu (Path): Đường dẫn lưu PNG.
        top (int): Số biến hiển thị.

    Trả ra:
        pd.DataFrame: Top biến theo Mutual Information.
    """
    bo = {'Target', 'Target_Graduate'}
    cot = [c for c in du_lieu.columns if c not in bo and not c.startswith('Curricular_units')]
    X = du_lieu[cot].apply(pd.to_numeric, errors='coerce').fillna(0)
    mi = pd.Series(mutual_info_classif(X, du_lieu['Target_Graduate'], random_state=42), index=[c.lstrip('﻿') for c in cot])
    mi = mi.sort_values(ascending=False).head(top)
    mau = ['#c53030' if 'Tuition' in c else ('#2b6cb0' if c in cac_bien_z else '#a0aec0') for c in mi.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(mi.index[::-1], mi.values[::-1], color=mau[::-1])
    ax.set_xlabel('Mutual Information với Y (tốt nghiệp)')
    ax.set_title('Hình 1.2. Liên kết của biến nền với kết cục — xanh: đang dùng làm T/Z, đỏ: học phí đúng hạn, xám: không dùng', fontsize=10)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()
    return mi.round(4).to_frame('Mutual_Information')


# ==============================================================================================================
# bang_overlap_theo_tang  (def)
# Vị trí: Cell 14 · mục 1.6 Kiểm toán dữ liệu bổ sung trước khi mô hình hóa
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Bảng 1.6b: đếm số SV có/không học bổng trong 6 tầng (3 nhóm điểm vào x 2 nhóm tuổi) để kiểm tra
#            overlap thô.
# CÁCH LÀM :
#   1. Tạo nhóm điểm (3 mức) và nhóm tuổi (>20 / <=20).
#   2. Đếm tổng số SV và số có học bổng trong từng tầng; suy ra số không học bổng và tỷ lệ.
#   3. Gắn cờ "tầng thưa" nếu một trong hai nhóm có < 30 SV.
# LÝ THUYẾT: Positivity: 0 < P(T=1 | Z=z) < 1 với mọi z. Thiếu một nhóm thì mô hình phải ngoại suy.
# ❓ HỎI   : Overlap có đạt không?
#   ➜ ĐÁP : Ở mức 6 tầng thì đạt (tầng nhỏ nhất vẫn có 44 SV có học bổng). Khi chia đủ 5 biến thì có vài hồ sơ
#           thiếu hỗ trợ, ghi ở cột Overlap_Status của Table B.
# ==============================================================================================================
def bang_overlap_theo_tang(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này để làm gì: Đếm số sinh viên có / không có học bổng trong từng tầng (nhóm điểm vào x nhóm tuổi),
    để kiểm tra overlap trước khi ước lượng (Slide 55: "inspect treatment counts within covariate groups").

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.

    Trả ra:
        pd.DataFrame: Một dòng mỗi tầng, kèm cờ cảnh báo tầng thưa.
    """
    d = du_lieu.assign(
        Nhom_diem=pd.cut(du_lieu['Admission_grade'], bins=[-np.inf, 120, 140, np.inf], labels=['< 120', '120–140', '≥ 140'], right=False),
        Nhom_tuoi=np.where(du_lieu['Age_at_enrollment'] > 20, '> 20', '≤ 20'))
    bang = d.groupby(['Nhom_diem', 'Nhom_tuoi'], observed=True).agg(
        So_SV=('Scholarship_holder', 'size'), Co_hoc_bong=('Scholarship_holder', 'sum'))
    bang['Khong_hoc_bong'] = bang['So_SV'] - bang['Co_hoc_bong']
    bang['Ty_le_hoc_bong'] = (bang['Co_hoc_bong'] / bang['So_SV']).round(3)
    bang['Canh_bao'] = np.where(bang[['Co_hoc_bong', 'Khong_hoc_bong']].min(axis=1) < 30, 'tầng thưa (< 30 ở một nhóm)', '')
    return bang.reset_index()
