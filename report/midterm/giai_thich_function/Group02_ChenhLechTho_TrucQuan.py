# -*- coding: utf-8 -*-
"""
NHÓM 02 — Chênh lệch quan sát thô và các hàm vẽ hình (mục 2, 3, 6, 8)

Tính P(Y|T) thô (+34.54 điểm %), vẽ Hình 2.1, và các hàm vẽ đồ thị có hướng dùng chung cho Hình 3.1, 6.1, 8.1
cùng forest plot ATE (Hình 8.3).

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - tinh_toan_do_chenh_lech_quan_sat_tho       Cell 17 (def)
  - ve_ty_le_tot_nghiep_theo_nhom              Cell 18 (def)
  - ve_do_thi_co_huong                         Cell 22 (def)
  - vi_tri_theo_tang                           Cell 31 (def)
  - do_cong_canh_z_z                           Cell 61 (def)
  - ve_forest_ate_table_b                      Cell 64 (def)
"""


# ==============================================================================================================
# tinh_toan_do_chenh_lech_quan_sat_tho  (def)
# Vị trí: Cell 17 · mục 2.2 Cơ chế Chính sách Cụ thể của Nhà trường ($do(T)$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính chênh lệch quan sát thô: P(Y=1|T=1) - P(Y=1|T=0) = 86.17% - 51.63% = +34.54 điểm %.
# CÁCH LÀM :
#   1. Lấy cột Target_Graduate của nhóm T=1 và T=0.
#   2. Lấy trung bình mỗi nhóm (= tỷ lệ tốt nghiệp).
#   3. Trừ hai tỷ lệ.
# LÝ THUYẾT: Đây là P(Y|T) (quan sát), KHÔNG phải P(Y|do(T)). Nó trộn hiệu ứng thật với nhiễu nền (SV giỏi dễ có
#            học bổng và dễ tốt nghiệp).
# ❓ HỎI   : Vậy cấp học bổng tăng tốt nghiệp 35% à?
#   ➜ ĐÁP : Không. Sau điều chỉnh nhiễu, ATE khoảng +24%; khoảng 10 điểm % là thiên lệch do nhiễu quan sát được.
# ==============================================================================================================
def tinh_toan_do_chenh_lech_quan_sat_tho(du_lieu: pd.DataFrame) -> Dict[str, float]:
    """
    Hàm này để làm gì: Tính toán tỷ lệ tốt nghiệp quan sát thô giữa nhóm có học bổng (T=1)
    và nhóm không có học bổng (T=0) để đo lường độ chênh lệch ban đầu.

    Nhận vào:
        du_lieu (pd.DataFrame): Bảng dữ liệu sinh viên nhị phân UCI.

    Trả ra:
        Dict[str, float]: Từ điển chứa tỷ lệ tốt nghiệp từng nhóm và độ chênh lệch thô (Naive Gap).
    """
    nhom_t1 = du_lieu[du_lieu['Scholarship_holder'] == 1]['Target_Graduate']
    nhom_t0 = du_lieu[du_lieu['Scholarship_holder'] == 0]['Target_Graduate']
    
    p_y_given_t1 = float(nhom_t1.mean())
    p_y_given_t0 = float(nhom_t0.mean())
    naive_gap = p_y_given_t1 - p_y_given_t0
    
    return {
        'P(Y=1|T=1)': p_y_given_t1,
        'P(Y=1|T=0)': p_y_given_t0,
        'Naive_Observational_Gap': naive_gap
    }


# ==============================================================================================================
# ve_ty_le_tot_nghiep_theo_nhom  (def)
# Vị trí: Cell 18 · mục 2.2 Cơ chế Chính sách Cụ thể của Nhà trường ($do(T)$)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 2.1: tỷ lệ tốt nghiệp theo học bổng (trái) và theo học bổng x học phí đúng hạn (phải), tô đỏ ô
#            thưa n < 100.
# CÁCH LÀM :
#   1. groupby Scholarship_holder: trung bình và số lượng -> cột trái.
#   2. groupby (Scholarship_holder, Tuition_fees_up_to_date) -> 4 cột phải; nhãn sinh tự động từ chỉ số groupby
#      nên không lệch.
#   3. Cột có n < 100 tô đỏ (ô T=1, chưa đóng đúng hạn chỉ có 37 SV).
# ❓ HỎI   : 24.3% ở cột đỏ có tin được không?
#   ➜ ĐÁP : Là 9/37; CI 95% Wilson khoảng [13%, 40%]. Ô này cho thấy positivity yếu, không dùng làm ước lượng
#           chính xác.
# ==============================================================================================================
def ve_ty_le_tot_nghiep_theo_nhom(du_lieu: pd.DataFrame, duong_dan_luu: Path) -> None:
    """
    Hàm này để làm gì: Vẽ tỷ lệ tốt nghiệp quan sát theo học bổng T, và theo tổ hợp (học bổng x học phí đúng hạn),
    ghi số sinh viên trên mỗi cột để thấy ngay các ô thưa (positivity).

    Nhận vào:
        du_lieu (pd.DataFrame): Quần thể nhị phân N=3,630.
        duong_dan_luu (Path): Đường dẫn lưu file PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1, 1.6]})

    nhom_t = du_lieu.groupby('Scholarship_holder')['Target_Graduate'].agg(['mean', 'size'])
    nhan_t = ['Không học bổng\n(T=0)', 'Có học bổng\n(T=1)']
    cot = ax1.bar(nhan_t, nhom_t['mean'] * 100, color=['#9aa5b1', '#2b6cb0'], width=0.55)
    for c, (ty_le, n) in zip(cot, nhom_t.values):
        ax1.text(c.get_x() + c.get_width() / 2, ty_le * 100 + 1.5, f'{ty_le*100:.1f}%\nn={int(n):,}', ha='center', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('Tỷ lệ tốt nghiệp quan sát (%)')
    ax1.set_title('P(Y=1 | T): chênh lệch thô +34.54 điểm %')

    nhom_tt = du_lieu.groupby(['Scholarship_holder', 'Tuition_fees_up_to_date'])['Target_Graduate'].agg(['mean', 'size'])
    nhan_tt = [f"T={t}\nHọc phí đúng hạn={f}" for t, f in nhom_tt.index]
    mau = ['#c53030' if n < 100 else '#4a5568' for n in nhom_tt['size']]
    cot2 = ax2.bar(nhan_tt, nhom_tt['mean'] * 100, color=mau, width=0.6)
    for c, (ty_le, n) in zip(cot2, nhom_tt.values):
        ax2.text(c.get_x() + c.get_width() / 2, ty_le * 100 + 1.5, f'{ty_le*100:.1f}%\nn={int(n):,}', ha='center', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.set_title('Theo học bổng x trạng thái học phí (đỏ = ô thưa, n < 100)')
    fig.suptitle('Hình 2.1. Phân bố kết cục Y theo can thiệp T (dữ liệu quan sát, chưa điều chỉnh nhiễu)', fontsize=11)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================================================
# ve_do_thi_co_huong  (def)
# Vị trí: Cell 22 · mục 3.2 Bảng so sánh tổng hợp các thuật toán học cấu trúc
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hàm vẽ đồ thị có hướng dùng chung cho Hình 3.1, 6.1, 8.1: vị trí nút cố định, tô đỏ / nét đứt các
#            cạnh được chỉ định.
# ĐẦU VÀO  : ax, danh sách cạnh, vị trí nút, nhãn nút, tiêu đề, tập cạnh đỏ, tập cạnh nét đứt, độ cong (số hoặc
#            dict theo cạnh), kích thước nút.
# CÁCH LÀM :
#   1. Tạo nx.DiGraph với đủ nút và cạnh.
#   2. Vẽ nút và nhãn.
#   3. Vẽ từng cạnh riêng: đỏ nếu thuộc canh_do, nét đứt nếu thuộc canh_net_dut; độ cong lấy từ do_cong (dict
#      thì tra theo cạnh, mặc định 0).
# LƯU Ý    : Tham số do_cong dạng dict được thêm để Hình 8.1 vẽ cạnh Z -> Z vòng ra ngoài, không đè lên nút
#            khác.
# ==============================================================================================================
def ve_do_thi_co_huong(ax: plt.Axes, cac_canh: List[Tuple[str, str]], vi_tri: Dict[str, Tuple[float, float]],
                       nhan: Dict[str, str], tieu_de: str, canh_do: Set[Tuple[str, str]],
                       canh_net_dut: Optional[Set[Tuple[str, str]]] = None,
                       do_cong: float | Dict[Tuple[str, str], float] = 0.08,
                       kich_thuoc_nut: int = 2600) -> None:
    """
    Hàm này để làm gì: Vẽ một đồ thị có hướng lên trục ax với vị trí nút cố định; tô đỏ các cạnh trong canh_do
    (ví dụ cạnh ngược thời gian) và vẽ nét đứt các cạnh trong canh_net_dut (ví dụ cạnh bị ép thêm/bị cắt).

    Nhận vào:
        ax (plt.Axes): Trục vẽ.
        cac_canh (List[Tuple[str, str]]): Danh sách cạnh (u, v).
        vi_tri (Dict[str, Tuple[float, float]]): Tọa độ từng nút.
        nhan (Dict[str, str]): Nhãn hiển thị của từng nút.
        tieu_de (str): Tiêu đề ô hình.
        canh_do (Set[Tuple[str, str]]): Các cạnh tô đỏ.
        canh_net_dut (Optional[Set[Tuple[str, str]]]): Các cạnh vẽ nét đứt.
        do_cong (float | Dict): Độ cong của cạnh (0 = thẳng); truyền dict {(u, v): độ cong} để cong riêng
            từng cạnh (cạnh không có trong dict vẽ thẳng).
        kich_thuoc_nut (int): Kích thước nút.
    """
    canh_net_dut = canh_net_dut or set()
    g = nx.DiGraph()
    g.add_nodes_from(vi_tri.keys())
    g.add_edges_from(cac_canh)
    nx.draw_networkx_nodes(g, vi_tri, ax=ax, node_size=kich_thuoc_nut, node_color='#edf2f7', edgecolors='#2d3748', linewidths=1.2)
    nx.draw_networkx_labels(g, vi_tri, labels=nhan, ax=ax, font_size=8.5)
    for (u, v) in g.edges():
        mau = '#c53030' if (u, v) in canh_do else '#2d3748'
        kieu = 'dashed' if (u, v) in canh_net_dut else 'solid'
        cong = do_cong.get((u, v), 0.0) if isinstance(do_cong, dict) else do_cong
        nx.draw_networkx_edges(g, vi_tri, edgelist=[(u, v)], ax=ax, edge_color=mau, style=kieu, width=2.0,
                               arrowsize=18, node_size=kich_thuoc_nut, connectionstyle=f'arc3,rad={cong}')
    ax.margins(0.12)
    ax.set_title(tieu_de, fontsize=10)
    ax.axis('off')


# ==============================================================================================================
# vi_tri_theo_tang  (def)
# Vị trí: Cell 31 · mục 6.1 Cấu trúc Đồ thị Phân cấp Thời gian (Primary Domain DAG với Biến Trung gian M1)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tính tọa độ nút cho Hình 6.1 theo tầng thời gian: Z cột trái, T và M1 hạ xuống hàng dưới, Y bên
#            phải.
# CÁCH LÀM :
#   1. Trục x = số tầng x 2.2.
#   2. Các nút cùng tầng xếp dọc cách đều quanh y = 0.
#   3. Tầng thuộc hang_duoi (T, M1) đặt ở y = -do_ha để cạnh Z -> Y không chạy xuyên qua T.
# LƯU Ý    : Chỉ phục vụ hiển thị, không ảnh hưởng tính toán.
# ==============================================================================================================
def vi_tri_theo_tang(tang: Dict[str, int], hang_duoi: Tuple[int, ...] = (2, 3),
                     do_ha: float = 3.2) -> Dict[str, Tuple[float, float]]:
    """
    Hàm này để làm gì: Tính tọa độ nút theo tầng thời gian (trục x = tầng). Các nút cùng tầng xếp dọc cách đều;
    các tầng trong hang_duoi (T và M1) được hạ xuống một hàng riêng bên dưới, để các cạnh Z -> Y đi thẳng
    sang Y mà không chạy xuyên qua nút T hay M1 (tránh các cạnh đè lên nhau).

    Nhận vào:
        tang (Dict[str, int]): Ánh xạ tên nút -> tầng thời gian.
        hang_duoi (Tuple[int, ...]): Các tầng được hạ xuống hàng dưới.
        do_ha (float): Tọa độ y (âm) của hàng dưới.

    Trả ra:
        Dict[str, Tuple[float, float]]: Tọa độ (x, y) của từng nút.
    """
    vi_tri: Dict[str, Tuple[float, float]] = {}
    for t in sorted(set(tang.values())):
        cac_nut = [n for n in tang if tang[n] == t]
        for k, n in enumerate(cac_nut):
            y = -do_ha if t in hang_duoi else (len(cac_nut) - 1) / 2.0 - k
            vi_tri[n] = (float(t) * 2.2, y)
    return vi_tri


# ==============================================================================================================
# do_cong_canh_z_z  (def)
# Vị trí: Cell 61 · mục 8.2b Giới hạn của $T_2, T_3$ (đọc bảng trên)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Gán độ cong cho cạnh nối hai biến Z cùng cột ở Hình 8.1: cạnh kề nhau vẽ thẳng, cạnh cách xa vòng
#            ra bên trái.
# CÁCH LÀM :
#   1. Chỉ xét cạnh có cả hai đầu là Z.
#   2. Số bước = khoảng cách dọc / 2; cạnh kề nhau (1 bước) để thẳng.
#   3. Cạnh xa: rad = (muc_cong + 0.7*(số bước - 2)) / |dy|, đổi dấu theo chiều đi lên hay xuống để luôn vòng
#      sang trái.
# LƯU Ý    : Chỉ phục vụ hiển thị.
# ==============================================================================================================
def do_cong_canh_z_z(cac_canh: List[Tuple[str, str]], vi_tri: Dict[str, Tuple[float, float]],
                     muc_cong: float = 3.0) -> Dict[Tuple[str, str], float]:
    """
    Hàm này để làm gì: Gán độ cong cho các cạnh nối hai biến Z cùng cột. Cạnh nối hai nút kề nhau vẽ thẳng;
    cạnh nối hai nút cách xa vòng ra phía bên trái (không chạy xuyên qua các nút Z nằm giữa), cạnh càng dài
    thì vòng càng rộng nên không đè lên nhau.

    Nhận vào:
        cac_canh (List[Tuple[str, str]]): Danh sách cạnh (u, v).
        vi_tri (Dict[str, Tuple[float, float]]): Tọa độ từng nút.
        muc_cong (float): Khoảng vòng ra bên trái (theo đơn vị trục x) của cạnh ngắn nhất cần cong.

    Trả ra:
        Dict[Tuple[str, str], float]: Độ cong của từng cạnh Z -> Z (cạnh khác vẽ thẳng).
    """
    cong: Dict[Tuple[str, str], float] = {}
    for (u, v) in cac_canh:
        if u.startswith('Z') and v.startswith('Z'):
            dy = vi_tri[v][1] - vi_tri[u][1]
            so_buoc = round(abs(dy) / 2)
            if so_buoc <= 1:
                continue
            rad = (muc_cong + 0.7 * (so_buoc - 2)) / abs(dy)
            cong[(u, v)] = rad if dy < 0 else -rad
    return cong


# ==============================================================================================================
# ve_forest_ate_table_b  (def)
# Vị trí: Cell 64 · mục 8.2c Đọc Hình 8.1 và bảng tóm tắt (B0 và H2 học được gì)
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Hình 8.3: forest plot ATE và CI 95% theo từng seed cho B0, H1, H2, kèm vạch chênh lệch thô.
# CÁCH LÀM :
#   1. Lấy các dòng theo seed (bỏ dòng tổng hợp), ép ATE/CI về số.
#   2. Mỗi dòng một điểm với thanh sai số (CI bootstrap); khoảng âm được chặn ở 0 để tránh lỗi khi CI suy biến.
#   3. Kẻ vạch xám tại chênh lệch thô +0.3454.
# ❓ HỎI   : Hình nói lên điều gì?
#   ➜ ĐÁP : Cả ba mô hình đều quanh +0.24, cách xa +0.345 thô: kết luận can thiệp không phụ thuộc mô hình.
# ==============================================================================================================
def ve_forest_ate_table_b(bang_b: pd.DataFrame, chenh_lech_tho: float, duong_dan_luu: Path) -> None:
    """
    Hàm này để làm gì: Vẽ forest plot ATE và 95% CI theo từng seed cho B0, H1, H2 (Table B), so với chênh lệch thô.

    Nhận vào:
        bang_b (pd.DataFrame): Table_B_Intervention.csv.
        chenh_lech_tho (float): Chênh lệch quan sát thô P(Y|T=1) - P(Y|T=0).
        duong_dan_luu (Path): Đường dẫn lưu PNG.
    """
    theo_seed = bang_b[bang_b['Seed'].astype(str).str.isdigit()].copy()
    for cot in ['ATE', 'CI_95_Low', 'CI_95_High']:
        theo_seed[cot] = theo_seed[cot].astype(float)
    mau = {'B0': '#2b6cb0', 'H1': '#2f855a', 'H2': '#c53030'}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = 0
    nhan_y: List[str] = []
    for mo_hinh in ['B0', 'H1', 'H2']:
        con = theo_seed[theo_seed['Model'] == mo_hinh]
        for _, r in con.iterrows():
            ax.errorbar(r['ATE'], y, xerr=[[max(r['ATE'] - r['CI_95_Low'], 0.0)], [max(r['CI_95_High'] - r['ATE'], 0.0)]],  # H2: CI suy biến
                        fmt='o', color=mau[mo_hinh], capsize=3)
            nhan_y.append(f"{mo_hinh} · seed {r['Seed']}")
            y += 1
        y += 0.6
        nhan_y.append('')
    ax.axvline(chenh_lech_tho, color='#718096', ls='--', lw=1.2)
    ax.text(chenh_lech_tho - 0.002, 1.0, f'chênh lệch thô {chenh_lech_tho:+.4f}', ha='right', fontsize=8.5, color='#4a5568')
    toa_do = []
    yy = 0
    for mo_hinh in ['B0', 'H1', 'H2']:
        for _ in range(len(theo_seed[theo_seed['Model'] == mo_hinh])):
            toa_do.append(yy)
            yy += 1
        yy += 0.6
    ax.set_yticks(toa_do)
    ax.set_yticklabels([n for n in nhan_y if n])
    ax.invert_yaxis()
    ax.set_xlabel('ATE = E[Y | do(T=1)] - E[Y | do(T=0)] trên C_test (n=545), 95% bootstrap CI')
    ax.set_title('Hình 8.3. ATE theo seed (Table B): B0, H1, H2 đều ≈ +0.24, xa chênh lệch thô', fontsize=11)
    fig.tight_layout()
    fig.savefig(duong_dan_luu, dpi=150, bbox_inches='tight')
    plt.show()
