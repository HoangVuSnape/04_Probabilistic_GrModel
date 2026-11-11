# -*- coding: utf-8 -*-
"""
NHÓM 03 — Minh họa học cấu trúc và học tham số (mục 3, 4)

HillClimb + BIC có / không ràng buộc (Hình 3.1) và so sánh CPD MLE với CPD BDeu.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - hoc_cau_truc_hill_climb_bic                Cell 21 (def)
  - so_sanh_hoc_tham_so_mle_va_bayesian        Cell 25 (def)
"""


# ==============================================================================================================
# hoc_cau_truc_hill_climb_bic  (def)
# Vị trí: Cell 21 · mục 3.2 Bảng so sánh tổng hợp các thuật toán học cấu trúc
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Minh họa (mục 3) HillClimb + BIC trên 4 biến, có hoặc không có ExpertKnowledge cấm cạnh.
# ĐẦU VÀO  : du_lieu (4 biến rời rạc), danh_sach_cam (None = học tự do), so_vong_lap_toi_da.
# ĐẦU RA   : Danh sách cạnh học được.
# CÁCH LÀM :
#   1. Tạo HillClimbSearch và hàm điểm BIC.
#   2. Nếu có danh sách cấm thì gói vào ExpertKnowledge(forbidden_edges=...).
#   3. estimate(): leo đồi bằng thêm/xóa/đảo cạnh, giữ thay đổi làm điểm tăng nhiều nhất.
# LÝ THUYẾT: BIC = log-likelihood - (số tham số / 2) x log N: phạt đồ thị phức tạp.
# ❓ HỎI   : Kết quả minh họa là gì?
#   ➜ ĐÁP : Học tự do ra cạnh ngược Y -> Z; cấm cạnh ra từ Y thì hết nhưng còn T -> Debtor vì chưa cấm T -> Z.
#           Bài học: ràng buộc phải đủ Z -> T -> Y, như ở B0 mục 8.B.
# ==============================================================================================================
def hoc_cau_truc_hill_climb_bic(
    du_lieu: pd.DataFrame,
    danh_sach_cam: Optional[List[Tuple[str, str]]] = None,
    so_vong_lap_toi_da: int = 25
) -> List[Tuple[str, str]]:
    """
    Hàm này để làm gì: Khảo sát thuật toán học cấu trúc Hill Climb Search kết hợp hàm điểm BIC
    và đối tượng ràng buộc ExpertKnowledge để tìm kiếm cấu trúc nhân quả hợp lệ.

    Nhận vào:
        du_lieu (pd.DataFrame): Bảng dữ liệu chứa các biến khảo sát.
        danh_sach_cam (Optional[List[Tuple[str, str]]]): Danh sách các cạnh bị cấm xuất hiện.
        so_vong_lap_toi_da (int): Số lượt lặp tối đa của thuật toán leo đồi.

    Trả ra:
        List[Tuple[str, str]]: Danh sách các cạnh có hướng (u, v) được thuật toán học ra.
    """
    estimator = HillClimbSearch(du_lieu)
    scoring = BIC(du_lieu)
    
    tri_thuc = None
    if danh_sach_cam is not None:
        tri_thuc = ExpertKnowledge(forbidden_edges=danh_sach_cam)
        
    mo_hinh_uoc_luong = estimator.estimate(
        scoring_method=scoring,
        expert_knowledge=tri_thuc,
        max_iter=so_vong_lap_toi_da,
        show_progress=False
    )
    
    return list(mo_hinh_uoc_luong.edges())


# ==============================================================================================================
# so_sanh_hoc_tham_so_mle_va_bayesian  (def)
# Vị trí: Cell 25 · mục 4.2 Bảng so sánh các phương pháp học tham số
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : So sánh CPD của Y ước lượng bằng MLE (đếm tần suất) và bằng Bayes với tiên nghiệm BDeu (ESS = 10).
# CÁCH LÀM :
#   1. Dựng mạng Bayes con với cạnh Học bổng -> Y, Debtor -> Y.
#   2. MLE: P(y | cha) = N(y, cha) / N(cha).
#   3. BDeu: P(y | cha) = (N(y, cha) + ESS/(r q)) / (N(cha) + ESS/q), r = số trạng thái của Y, q = số cấu hình
#      cha.
# LÝ THUYẾT: BDeu là tiên nghiệm Dirichlet đều, tương đương thêm ESS "mẫu ảo" chia đều cho các ô; ô càng thưa
#            càng bị kéo về đều.
# ❓ HỎI   : Vì sao dùng BDeu thay MLE?
#   ➜ ĐÁP : Ô thưa (ví dụ Debtor=1 và có học bổng) thì MLE dễ ra 0% hoặc 100%; BDeu làm mịn, tránh xác suất cực
#           đoan.
# ==============================================================================================================
def so_sanh_hoc_tham_so_mle_va_bayesian(
    du_lieu: pd.DataFrame,
    cac_canh: List[Tuple[str, str]],
    bien_muc_tieu: str,
    equivalent_sample_size: float = 10.0
) -> Tuple[str, str]:
    """
    Hàm này để làm gì: So sánh tham số phân phối điều kiện xác suất (CPD) được ước lượng
    bởi phương pháp Cực đại Hợp lý (MLE) và phương pháp Bayes với tiền nghiệm BDeu.

    Nhận vào:
        du_lieu (pd.DataFrame): Bảng dữ liệu rời rạc mẫu.
        cac_canh (List[Tuple[str, str]]): Cấu trúc đồ thị xác định trước.
        bien_muc_tieu (str): Tên nút cần trích xuất bảng phân phối CPD.
        equivalent_sample_size (float): Cỡ mẫu tương đương (ESS) cho tiền nghiệm Dirichlet.

    Trả ra:
        Tuple[str, str]: Chuỗi biểu diễn bảng CPD của MLE và Bayesian BDeu.
    """
    mo_hinh = DiscreteBayesianNetwork(cac_canh)
    
    # 1. Ước lượng bằng MLE
    estimator_mle = MaximumLikelihoodEstimator(mo_hinh, du_lieu)
    cpd_mle = estimator_mle.estimate_cpd(bien_muc_tieu)
    
    # 2. Ước lượng bằng Bayesian Estimator (BDeu)
    estimator_bayes = BayesianEstimator(mo_hinh, du_lieu)
    cpd_bayes = estimator_bayes.estimate_cpd(
        bien_muc_tieu,
        prior_type="BDeu",
        equivalent_sample_size=equivalent_sample_size
    )
    
    return str(cpd_mle), str(cpd_bayes)
