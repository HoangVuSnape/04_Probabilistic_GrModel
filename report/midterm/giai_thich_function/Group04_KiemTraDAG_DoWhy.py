# -*- coding: utf-8 -*-
"""
NHÓM 04 — Bốn kiểm tra hợp lệ của DAG và quy trình DoWhy trên Domain DAG (mục 5, 6, 7)

Kiểm tra phi chu trình, phân tầng thời gian, danh sách cạnh cấm/bắt buộc, tiêu chuẩn cửa sau; chạy DoWhy 4
bước (model, identify, estimate, refute) trên Domain DAG.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - KetQuaKiemTraRangBuoc                      Cell 28 (class)
  - kiem_tra_tinh_khong_chu_trinh              Cell 28 (def)
  - kiem_tra_phan_tang_thoi_gian               Cell 28 (def)
  - kiem_tra_danh_sach_chuyen_gia              Cell 28 (def)
  - kiem_tra_d_separation_backdoor             Cell 28 (def)
  - KetQuaUocLuongDoWhy                        Cell 35 (class)
  - KetQuaKiemDinhRefute                       Cell 35 (class)
  - thuc_thi_quy_trinh_can_thiep_dowhy         Cell 35 (def)
"""


# ==============================================================================================================
# KetQuaKiemTraRangBuoc  (class)
# Vị trí: Cell 28 · mục 5.1 Định nghĩa 4 nhóm ràng buộc kiểm định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass kết quả của một bài kiểm tra: có hợp lệ không, tên bài kiểm tra, danh sách vi phạm.
# ==============================================================================================================
@dataclass
class KetQuaKiemTraRangBuoc:
    hop_le: bool
    ten_kiem_tra: str
    danh_sach_vi_pham: List[str]


# ==============================================================================================================
# kiem_tra_tinh_khong_chu_trinh  (def)
# Vị trí: Cell 28 · mục 5.1 Định nghĩa 4 nhóm ràng buộc kiểm định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Kiểm tra đồ thị có phải DAG (không có chu trình) không; nếu không thì liệt kê chu trình.
# CÁCH LÀM :
#   1. nx.is_directed_acyclic_graph.
#   2. Nếu có chu trình, liệt kê bằng nx.simple_cycles.
# LÝ THUYẾT: Mạng Bayes chỉ phân rã P(X) = prod P(X_i | Pa(X_i)) được khi đồ thị không có chu trình.
# ==============================================================================================================
def kiem_tra_tinh_khong_chu_trinh(do_thi: nx.DiGraph) -> KetQuaKiemTraRangBuoc:
    """
    Hàm này để làm gì: Kiểm tra xem đồ thị có hướng có thỏa mãn tính chất phi chu trình (DAG) hay không.

    Nhận vào:
        do_thi (nx.DiGraph): Đồ thị có hướng cần kiểm tra.

    Trả ra:
        KetQuaKiemTraRangBuoc: Trạng thái hợp lệ và chi tiết chu trình vi phạm nếu có.
    """
    la_dag = nx.is_directed_acyclic_graph(do_thi)
    vi_pham: List[str] = []
    if not la_dag:
        try:
            chu_trinh = list(nx.simple_cycles(do_thi))
            vi_pham = [f"Phát hiện chu trình: {' -> '.join(c)} -> {c[0]}" for c in chu_trinh]
        except Exception:
            vi_pham = ["Đồ thị chứa ít nhất một chu trình có hướng."]
            
    return KetQuaKiemTraRangBuoc(hop_le=la_dag, ten_kiem_tra="Acyclicity (Tính phi chu trình)", danh_sach_vi_pham=vi_pham)


# ==============================================================================================================
# kiem_tra_phan_tang_thoi_gian  (def)
# Vị trí: Cell 28 · mục 5.1 Định nghĩa 4 nhóm ràng buộc kiểm định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Kiểm tra không có cạnh đi ngược thời gian: với mỗi cạnh u -> v, tầng(u) phải <= tầng(v).
# CÁCH LÀM :
#   1. Duyệt mọi cạnh, tra tầng của hai đầu trong bang_phan_tang (Z=1, T=2, M1=3, Y=4).
#   2. Cạnh có tầng nguồn lớn hơn tầng đích là vi phạm.
# ❓ HỎI   : Cạnh cùng tầng có bị cấm không?
#   ➜ ĐÁP : Không (điều kiện là >, không phải >=); chiều giữa các Z cùng tầng để dữ liệu/chuyên gia quyết.
# ==============================================================================================================
def kiem_tra_phan_tang_thoi_gian(do_thi: nx.DiGraph, bang_phan_tang: Dict[str, int]) -> KetQuaKiemTraRangBuoc:
    """
    Hàm này để làm gì: Kiểm tra xem đồ thị có chứa cạnh nào vi phạm trình tự thời gian hành chính hay không.

    Nhận vào:
        do_thi (nx.DiGraph): Đồ thị có hướng cần kiểm định.
        bang_phan_tang (Dict[str, int]): Ánh xạ từ tên nút sang chỉ số tầng thời gian (1, 2, 3, 4).

    Trả ra:
        KetQuaKiemTraRangBuoc: Trạng thái hợp lệ và danh sách các cạnh ngược thời gian nếu có.
    """
    vi_pham: List[str] = []
    for u, v in do_thi.edges():
        tier_u = bang_phan_tang.get(u, -1)
        tier_v = bang_phan_tang.get(v, -1)
        if tier_u > tier_v:
            vi_pham.append(f"Cạnh ngược thời gian: {u} (Tầng {tier_u}) --> {v} (Tầng {tier_v})")
            
    return KetQuaKiemTraRangBuoc(
        hop_le=len(vi_pham) == 0,
        ten_kiem_tra="Temporal Tier (Phân tầng thời gian)",
        danh_sach_vi_pham=vi_pham
    )


# ==============================================================================================================
# kiem_tra_danh_sach_chuyen_gia  (def)
# Vị trí: Cell 28 · mục 5.1 Định nghĩa 4 nhóm ràng buộc kiểm định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Kiểm tra blacklist (cạnh cấm không được có) và whitelist (cạnh bắt buộc phải có).
# CÁCH LÀM :
#   1. Mỗi cạnh trong danh_sach_cam mà có mặt -> vi phạm.
#   2. Mỗi cạnh trong danh_sach_bat_buoc mà vắng mặt -> vi phạm.
# ==============================================================================================================
def kiem_tra_danh_sach_chuyen_gia(
    do_thi: nx.DiGraph,
    danh_sach_cam: List[Tuple[str, str]],
    danh_sach_bat_buoc: List[Tuple[str, str]]
) -> KetQuaKiemTraRangBuoc:
    """
    Hàm này để làm gì: Kiểm định đồ thị dựa trên danh sách cạnh cấm (Blacklist) và cạnh bắt buộc (Whitelist).

    Nhận vào:
        do_thi (nx.DiGraph): Đồ thị có hướng cần kiểm định.
        danh_sach_cam (List[Tuple[str, str]]): Các cạnh tuyệt đối không được xuất hiện.
        danh_sach_bat_buoc (List[Tuple[str, str]]): Các cạnh bắt buộc phải hiện diện.

    Trả ra:
        KetQuaKiemTraRangBuoc: Trạng thái hợp lệ và các điểm vi phạm cụ thể.
    """
    vi_pham: List[str] = []
    tap_canh = set(do_thi.edges())
    
    # Kiểm tra blacklist
    for u, v in danh_sach_cam:
        if (u, v) in tap_canh:
            vi_pham.append(f"Vi phạm Blacklist: Cạnh cấm ({u} --> {v}) lại xuất hiện trong đồ thị.")
            
    # Kiểm tra whitelist
    for u, v in danh_sach_bat_buoc:
        if (u, v) not in tap_canh:
            vi_pham.append(f"Vi phạm Whitelist: Thiếu cạnh bắt buộc ({u} --> {v}).")
            
    return KetQuaKiemTraRangBuoc(
        hop_le=len(vi_pham) == 0,
        ten_kiem_tra="Domain Constraints (Ràng buộc chuyên gia)",
        danh_sach_vi_pham=vi_pham
    )


# ==============================================================================================================
# kiem_tra_d_separation_backdoor  (def)
# Vị trí: Cell 28 · mục 5.1 Định nghĩa 4 nhóm ràng buộc kiểm định
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Kiểm tra tập Z có thỏa tiêu chuẩn cửa sau không: chặn mọi đường cửa sau giữa T và Y.
# CÁCH LÀM :
#   1. Tạo đồ thị G gạch dưới T: xóa mọi cạnh ĐI RA từ T (giữ lại đường đi VÀO T, tức đường cửa sau).
#   2. Dùng pgmpy DAG.is_dconnected(T, Y, observed=Z): nếu không còn d-connected thì Z chặn hết cửa sau.
# LÝ THUYẾT: Tiêu chuẩn cửa sau (Pearl): Z không chứa con cháu của T và T d-tách Y khi biết Z trong G gạch dưới
#            T. Khi đó P(y|do(t)) = sum_z P(y|t,z) P(z).
# ❓ HỎI   : Vì sao xóa cạnh đi ra từ T?
#   ➜ ĐÁP : Để chỉ còn lại các đường không nhân quả (cửa sau); đường nhân quả T -> ... -> Y không được chặn.
# ==============================================================================================================
def kiem_tra_d_separation_backdoor(
    do_thi: nx.DiGraph,
    nut_can_thiep: str,
    nut_ket_qua: str,
    tap_dieu_kien: Set[str]
) -> KetQuaKiemTraRangBuoc:
    """
    Hàm này để làm gì: Kiểm tra xem tập biến điều kiện có thỏa mãn tính chất d-separation
    đối với tất cả các đường dẫn cửa sau (backdoor paths) hay không.

    Nhận vào:
        do_thi (nx.DiGraph): Đồ thị nhân quả miền.
        nut_can_thiep (str): Nút can thiệp T.
        nut_ket_qua (str): Nút kết quả Y.
        tap_dieu_kien (Set[str]): Tập các biến quan sát được kiểm soát Z.

    Trả ra:
        KetQuaKiemTraRangBuoc: Trạng thái hợp lệ của tiêu chuẩn cửa sau.
    """
    from pgmpy.base import DAG
    
    # Đồ thị G_underline_T: Xóa toàn bộ các mũi tên đi ra từ nut_can_thiep
    do_thi_underline = do_thi.copy()
    cac_canh_ra = list(do_thi_underline.out_edges(nut_can_thiep))
    do_thi_underline.remove_edges_from(cac_canh_ra)
    
    # Kiểm tra d-separation trên đồ thị phẫu thuật thông qua pgmpy DAG
    p_dag = DAG(list(do_thi_underline.edges()))
    la_d_separated = not p_dag.is_dconnected(nut_can_thiep, nut_ket_qua, observed=tap_dieu_kien)
    
    vi_pham: List[str] = []
    if not la_d_separated:
        vi_pham.append(f"Tập điều kiện Z={tap_dieu_kien} KHÔNG chặn hết các đường dẫn backdoor nối {nut_can_thiep} và {nut_ket_qua}!")
        
    return KetQuaKiemTraRangBuoc(
        hop_le=la_d_separated,
        ten_kiem_tra="Backdoor Admissibility (Tiêu chuẩn cửa sau)",
        danh_sach_vi_pham=vi_pham
    )


# ==============================================================================================================
# KetQuaUocLuongDoWhy  (class)
# Vị trí: Cell 35 · mục 7.1 Khung quy trình 4 bước của DoWhy (Model - Identify - Estimate - Refute)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass kết quả ước lượng DoWhy: tên phương pháp, ATE, tập biến cửa sau.
# ==============================================================================================================
@dataclass
class KetQuaUocLuongDoWhy:
    phuong_phap: str
    gia_tri_ate: float
    cac_bien_backdoor: List[str]


# ==============================================================================================================
# KetQuaKiemDinhRefute  (class)
# Vị trí: Cell 35 · mục 7.1 Khung quy trình 4 bước của DoWhy (Model - Identify - Estimate - Refute)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass kết quả một refuter: tên, hiệu ứng gốc, hiệu ứng sau bác bỏ, kết luận.
# ==============================================================================================================
@dataclass
class KetQuaKiemDinhRefute:
    ten_kiem_dinh: str
    hieu_ung_goc: float
    hieu_ung_sau_bac_bo: float
    ket_luan: str


# ==============================================================================================================
# thuc_thi_quy_trinh_can_thiep_dowhy  (def)
# Vị trí: Cell 35 · mục 7.1 Khung quy trình 4 bước của DoWhy (Model - Identify - Estimate - Refute)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Chạy đủ 4 bước DoWhy trên Domain DAG (mục 7): model -> identify -> estimate (2 cách) -> refute (3
#            cách).
# ĐẦU VÀO  : Dữ liệu 3,630 SV (biến gốc), tên T, tên Y, đồ thị dạng chuỗi DOT.
# ĐẦU RA   : (model, kết quả hồi quy tuyến tính, kết quả propensity score weighting, danh sách 3 refuter).
# CÁCH LÀM :
#   1. CausalModel(data, treatment, outcome, graph).
#   2. identify_effect: DoWhy tự tìm tập cửa sau (5 biến Z).
#   3. estimate_effect "backdoor.linear_regression": hồi quy Y ~ T + Z, hệ số của T là ATE (+24.12%).
#   4. estimate_effect "backdoor.propensity_score_weighting": trọng số nghịch đảo xác suất nhận học bổng
#      (+25.16%).
#   5. Refuter 1 random_common_cause: thêm một biến nhiễu ngẫu nhiên -> ATE không được đổi (lệch < 0.02).
#   6. Refuter 2 placebo_treatment_refuter: thay T bằng biến giả -> ATE phải về gần 0 (< 0.03).
#   7. Refuter 3 data_subset_refuter 80%: ước lượng lại trên 80% dữ liệu -> ATE ổn định (lệch < 0.02).
# LƯU Ý    : np.random.seed(42) trước mỗi refuter vì DoWhy vẫn dùng bộ sinh số toàn cục của numpy; nhờ vậy số
#            tái lập được.
# ❓ HỎI   : Refuter qua thì chứng minh không có nhiễu ẩn à?
#   ➜ ĐÁP : Không. Refuter chỉ kiểm tra độ ổn định của bộ ước lượng; nhiễu ẩn được đánh giá bằng RV ở mục 9.H.
# ==============================================================================================================
def thuc_thi_quy_trinh_can_thiep_dowhy(
    du_lieu: pd.DataFrame,
    bien_can_thiep: str,
    bien_ket_qua: str,
    chuoi_do_thi_dot: str
) -> Tuple[CausalModel, KetQuaUocLuongDoWhy, KetQuaUocLuongDoWhy, List[KetQuaKiemDinhRefute]]:
    """
    Hàm này để làm gì: Thực hiện trọn vẹn quy trình 4 bước của thư viện DoWhy:
    Khởi tạo mô hình -> Nhận dạng Backdoor -> Ước lượng ATE (Hồi quy và PSW) -> 3 Kiểm định bác bỏ.

    Nhận vào:
        du_lieu (pd.DataFrame): Bảng dữ liệu thực tế sinh viên UCI.
        bien_can_thiep (str): Tên biến can thiệp chính sách (Scholarship_holder).
        bien_ket_qua (str): Tên biến kết quả tốt nghiệp (Target_Graduate).
        chuoi_do_thi_dot (str): Chuỗi DOT mô tả cấu trúc Primary Domain DAG.

    Trả ra:
        Tuple[CausalModel, KetQuaUocLuongDoWhy, KetQuaUocLuongDoWhy, List[KetQuaKiemDinhRefute]]:
            - model: Đối tượng mô hình CausalModel đã gắn đồ thị.
            - kq_lr: Kết quả ước lượng ATE bằng Backdoor Linear Regression.
            - kq_psw: Kết quả ước lượng ATE bằng Propensity Score Weighting.
            - danh_sach_bac_bo: Kết quả 3 bài kiểm định bác bỏ đối kháng.
    """
    # Bước 1: Khởi tạo mô hình
    model = CausalModel(
        data=du_lieu,
        treatment=bien_can_thiep,
        outcome=bien_ket_qua,
        graph=chuoi_do_thi_dot
    )
    
    # Bước 2: Nhận dạng Backdoor
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    backdoor_vars = identified_estimand.get_backdoor_variables()
    
    # Bước 3: Ước lượng ATE bằng Linear Regression
    est_lr = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    kq_lr = KetQuaUocLuongDoWhy(
        phuong_phap="Backdoor Linear Regression",
        gia_tri_ate=float(est_lr.value),
        cac_bien_backdoor=backdoor_vars
    )
    
    # Bước 3b: Ước lượng ATE bằng Propensity Score Weighting
    est_psw = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting")
    kq_psw = KetQuaUocLuongDoWhy(
        phuong_phap="Propensity Score Weighting",
        gia_tri_ate=float(est_psw.value),
        cac_bien_backdoor=backdoor_vars
    )
    
    # Bước 4: 3 bài kiểm định bác bỏ (Refutations)
    danh_sach_bac_bo: List[KetQuaKiemDinhRefute] = []
    
    # 4.1. Random Common Cause Refuter
    np.random.seed(42)  # DoWhy còn dùng bộ sinh số toàn cục của numpy
    ref_common = model.refute_estimate(identified_estimand, est_lr, method_name="random_common_cause", num_simulations=5, random_state=42)
    danh_sach_bac_bo.append(KetQuaKiemDinhRefute(
        ten_kiem_dinh="Random Common Cause",
        hieu_ung_goc=float(est_lr.value),
        hieu_ung_sau_bac_bo=float(ref_common.new_effect),
        ket_luan="VƯỢT QUA (Hiệu ứng không đổi)" if abs(ref_common.new_effect - est_lr.value) < 0.02 else "THẤT BẠI"
    ))
    
    # 4.2. Placebo Treatment Refuter
    np.random.seed(42)  # DoWhy còn dùng bộ sinh số toàn cục của numpy
    ref_placebo = model.refute_estimate(identified_estimand, est_lr, method_name="placebo_treatment_refuter", num_simulations=5, random_state=42)
    danh_sach_bac_bo.append(KetQuaKiemDinhRefute(
        ten_kiem_dinh="Placebo Treatment Refuter",
        hieu_ung_goc=float(est_lr.value),
        hieu_ung_sau_bac_bo=float(ref_placebo.new_effect),
        ket_luan="VƯỢT QUA (Hiệu ứng tiệm cận 0)" if abs(ref_placebo.new_effect) < 0.03 else "THẤT BẠI"
    ))
    
    # 4.3. Data Subset Refuter (80% sample)
    np.random.seed(42)  # DoWhy còn dùng bộ sinh số toàn cục của numpy
    ref_subset = model.refute_estimate(identified_estimand, est_lr, method_name="data_subset_refuter", num_simulations=5, subset_fraction=0.8, random_state=42)
    danh_sach_bac_bo.append(KetQuaKiemDinhRefute(
        ten_kiem_dinh="Data Subset Refuter (80%)",
        hieu_ung_goc=float(est_lr.value),
        hieu_ung_sau_bac_bo=float(ref_subset.new_effect),
        ket_luan="VƯỢT QUA (Hiệu ứng ổn định)" if abs(ref_subset.new_effect - est_lr.value) < 0.02 else "THẤT BẠI"
    ))
    
    return model, kq_lr, kq_psw, danh_sach_bac_bo
