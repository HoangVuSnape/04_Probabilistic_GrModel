# -*- coding: utf-8 -*-
"""Chú thích chi tiết cho từng hàm/class trong notebook chính (dùng để sinh các file GroupNN_*.py)."""

# Mỗi nhóm: (số, tên file, tiêu đề, mô tả nhóm, danh sách mục)
# Mỗi mục: ('fn', tên) = hàm/class trích nguyên văn; ('code', cell, tên_khối) = đoạn mã tuần tự của cell (bọc trong hàm để đọc)
GROUPS = [
    (1, 'NapDuLieu_KiemToan', 'Nạp dữ liệu UCI 697 và kiểm toán trước khi mô hình hóa (mục 1)',
     'Đọc dữ liệu gốc, lọc quần thể Graduate/Dropout, ghi môi trường chạy, báo cáo luồng loại trừ, '
     'phân loại biến theo thời điểm đo, vẽ EDA (Hình 1.1, 1.2) và kiểm tra overlap thô (Bảng 1.6b).',
     [('fn', '_dinh_dang_so'), ('fn', 'ThongKeTapDuLieu'), ('fn', 'doc_va_kiem_toan_du_lieu_uci'),
      ('fn', 'in_thong_tin_moi_truong'), ('fn', 'luong_loai_tru'), ('fn', 'kiem_toan_thoi_diem_bien'),
      ('fn', 've_phan_bo_diem_vao'), ('fn', 've_mutual_information'), ('fn', 'bang_overlap_theo_tang')]),
    (2, 'ChenhLechTho_TrucQuan', 'Chênh lệch quan sát thô và các hàm vẽ hình (mục 2, 3, 6, 8)',
     'Tính P(Y|T) thô (+34.54 điểm %), vẽ Hình 2.1, và các hàm vẽ đồ thị có hướng dùng chung cho '
     'Hình 3.1, 6.1, 8.1 cùng forest plot ATE (Hình 8.3).',
     [('fn', 'tinh_toan_do_chenh_lech_quan_sat_tho'), ('fn', 've_ty_le_tot_nghiep_theo_nhom'),
      ('fn', 've_do_thi_co_huong'), ('fn', 'vi_tri_theo_tang'), ('fn', 'do_cong_canh_z_z'),
      ('fn', 've_forest_ate_table_b')]),
    (3, 'HocCauTruc_HocThamSo', 'Minh họa học cấu trúc và học tham số (mục 3, 4)',
     'HillClimb + BIC có / không ràng buộc (Hình 3.1) và so sánh CPD MLE với CPD BDeu.',
     [('fn', 'hoc_cau_truc_hill_climb_bic'), ('fn', 'so_sanh_hoc_tham_so_mle_va_bayesian')]),
    (4, 'KiemTraDAG_DoWhy', 'Bốn kiểm tra hợp lệ của DAG và quy trình DoWhy trên Domain DAG (mục 5, 6, 7)',
     'Kiểm tra phi chu trình, phân tầng thời gian, danh sách cạnh cấm/bắt buộc, tiêu chuẩn cửa sau; '
     'chạy DoWhy 4 bước (model, identify, estimate, refute) trên Domain DAG.',
     [('fn', 'KetQuaKiemTraRangBuoc'), ('fn', 'kiem_tra_tinh_khong_chu_trinh'), ('fn', 'kiem_tra_phan_tang_thoi_gian'),
      ('fn', 'kiem_tra_danh_sach_chuyen_gia'), ('fn', 'kiem_tra_d_separation_backdoor'),
      ('fn', 'KetQuaUocLuongDoWhy'), ('fn', 'KetQuaKiemDinhRefute'), ('fn', 'thuc_thi_quy_trinh_can_thiep_dowhy')]),
    (5, 'CauHinh_DuLieu_Split', 'Cấu hình chung, mã hóa 7 biến và 5 split dùng chung (mục 8.A)',
     'Mọi siêu tham số của B0/H1/H2 nằm trong Config; dữ liệu được mã hóa thành Z1..Z5, T, Y; '
     '5 split 70/15/15 phân tầng được lưu ra JSON và đóng băng.',
     [('fn', 'Config'), ('fn', 'init_directories'), ('fn', 'set_seed'), ('fn', 'load_and_preprocess_uci697'),
      ('fn', 'get_or_create_splits')]),
    (6, 'MoHinh_B0', 'Mô hình B0: học DAG truyền thống + CPD BDeu + truy vấn can thiệp (mục 8.B)',
     'Ràng buộc thời gian, HillClimb với 3 hàm điểm chọn trên VALIDATION, khớp CPD BDeu, '
     'tính log-likelihood, và hàm truy vấn q_m(t, C_i) dùng chung cho B0/H1/H2.',
     [('fn', 'build_b0_dag'), ('fn', 'temporal_forbidden_edges'), ('fn', 'fit_discrete_bayesian_model'),
      ('fn', 'fit_bn_on_dag'), ('fn', 'validation_log_likelihood'), ('fn', 'learn_b0_dag'),
      ('fn', 'save_graph_artifacts'), ('fn', 'query_model_effects')]),
    (7, 'MoHinh_H1', 'Mô hình H1: thay CPD của Y bằng mạng nơ-ron trên DAG của B0 (mục 8.C)',
     'MLP PyTorch ước lượng P(Y=1 | Pa(Y)), huấn luyện có early stopping, lưu checkpoint, '
     'rồi chuyển MLP thành bảng CPD để gắn lại vào mạng Bayes.',
     [('fn', 'NeuralCPDModule'), ('fn', 'encode_h1_features'), ('fn', 'h1_input_dim'), ('fn', 'train_h1_mlp'),
      ('fn', 'save_h1_checkpoint'), ('fn', 'build_h1_bayesian_network')]),
    (8, 'MoHinh_H2', 'Mô hình H2: học DAG bằng NOTEARS-MLP có mặt nạ M (mục 8.D)',
     'Mặt nạ cạnh cấm, mạng NOTEARS phi tuyến, tối ưu augmented Lagrangian, '
     'lọc ngưỡng tau thành DAG và chọn tau trên VALIDATION.',
     [('fn', 'build_domain_mask'), ('fn', 'NotearsMLP'), ('fn', 'train_notears_mlp'), ('fn', 'graph_from_weights'),
      ('fn', 'select_h2_graph')]),
    (9, 'DanhGia_5Run_TablesAB', 'Đánh giá dùng chung, vòng lặp 5 run ghép cặp, xuất Table A và B (mục 8.E)',
     'Chỉ số dự báo (ngưỡng chọn trên VAL), CI bootstrap của ATE, định danh DoWhy trên DAG của từng mô hình, '
     'kiểm tra overlap, vòng lặp 5 seed và tổng hợp Table A, Table B.',
     [('fn', 'evaluate_prediction_metrics'), ('fn', 'compute_bootstrap_ate_ci'), ('fn', 'identify_with_dowhy'),
      ('fn', 'overlap_report'), ('fn', 'record_model_run'), ('code', 52, 'vong_lap_5_run_ghep_cap'),
      ('fn', 'mean_std'), ('code', 55, 'xuat_table_a_b')]),
    (10, 'HuanLuyen_Overfit', 'Quá trình huấn luyện, kiểm tra overfit, có / không early stopping (mục 8.E2)',
     'Vẽ đường cong loss của H1 và hội tụ của H2 (Hình 8.2), bảng AUROC/log-loss trên TRAIN/VAL/TEST '
     '(Bảng 8.E2), huấn luyện H1 không dừng sớm để so sánh và lưu 2 checkpoint (Hình 8.2b, Bảng 8.E2b).',
     [('fn', 've_qua_trinh_huan_luyen'), ('fn', 'bang_overfit_theo_tap'), ('fn', 'so_sanh_early_stopping'),
      ('fn', 've_early_stopping')]),
    (11, 'Refuter_BangKetQua', 'Refuter (Table C), hiển thị bảng, positivity, đọc lại đồ thị và q_m minh bạch (mục 8.F, 8.2b, 8.2c)',
     'Refuter DoWhy trên DAG của B0/H2, kiểm định huấn luyện lại cho H1, đọc và hiển thị các bảng, '
     'bảng 4 ô học bổng x học phí, đọc DAG đã lưu cho Hình 8.1 và tính tay q_m(t, C_i).',
     [('fn', 'dowhy_refute_on_graph'), ('fn', 'h1_retraining_refuter'), ('code', 57, 'chay_refuter_va_table_c'), ('fn', 'doc_bang_csv'), ('fn', 'bang_hien_thi'),
      ('fn', 'bang_positivity_T1_T2'), ('fn', 'doc_do_thi_da_luu'), ('fn', 'mat_na_thoi_gian'),
      ('fn', 'bang_tom_tat_do_thi'), ('fn', 'ma_hoa_7_bien'), ('fn', 'tai_mlp_h1'), ('fn', 'ma_hoa_dau_vao_h1'),
      ('fn', 'q_b0'), ('fn', 'q_h1')]),
    (12, 'PhanTichMoRong_Muc9', 'Phân tích mở rộng: kết cục khác, AIPW, trung gian, CATE, độ nhạy RV (mục 9)',
     'Mục 9 viết dạng tuần tự (không bọc hàm) cho 9.B–9.G; ở file này mỗi đoạn được bọc thành một hàm chỉ để đọc. '
     '9.H có hai hàm thật: tính Robustness Value và partial R^2 làm mốc.',
     [('fn', 'AdvancedConfig'), ('code', 69, 'nap_hai_quan_the'), ('code', 71, 'diem_hk2_ols'),
      ('code', 73, 'ty_le_dat_tin_chi_ols'), ('code', 75, 'giu_chan_aipw'), ('code', 77, 'trung_gian_nde_nie'),
      ('code', 79, 'cate_theo_nhom'), ('fn', 'KetQuaDoNhayCinelli'), ('fn', 'tinh_rv_cinelli_hazlett'),
      ('fn', 'benchmark_partial_r2')]),
    (13, 'PhanBoChinhSach_Muc10', 'Mô phỏng phân bổ học bổng theo CATE (mục 10)',
     'Mã hóa biến nền và so sánh 4 cách phân bổ cùng số suất học bổng bằng T-learner logistic (in-sample).',
     [('fn', 'ma_hoa_z_chinh_sach'), ('fn', 'danh_gia_chinh_sach')]),
]

# Chú thích: dict tên -> các trường. Trường nào thiếu thì bỏ qua.
A = {}

A['_dinh_dang_so'] = dict(
    muc='Định dạng số khi HIỂN THỊ bảng pandas: 4 chữ số thập phân, số nguyên in gọn, số rất nhỏ (p-value) dạng khoa học.',
    cach=['NaN thì in "NaN" (x != x chỉ đúng với NaN).',
          'Số nguyên (ví dụ 4424.0) in "4424".',
          'Số khác 0 nhưng nhỏ hơn 1e-4 (p-value 4.4e-41) in dạng khoa học để không bị thành 0.0000.',
          'Còn lại in 4 chữ số thập phân.'],
    luu_y='Được đăng ký bằng pd.set_option("display.float_format", ...). Chỉ đổi cách hiển thị; file CSV vẫn đủ độ chính xác.')

A['ThongKeTapDuLieu'] = dict(
    muc='Dataclass gom các con số tóm tắt quần thể: số mẫu, số tốt nghiệp/bỏ học, số nhận học bổng và tỷ lệ.',
    luu_y='Chỉ là "hộp chứa" dữ liệu, không có logic.')

A['doc_va_kiem_toan_du_lieu_uci'] = dict(
    muc='Đọc file UCI 697, chuẩn hóa tên cột, lọc quần thể Graduate/Dropout và tạo nhãn nhị phân Target_Graduate.',
    vao='duong_dan_tep: đường dẫn data/uci_697/data.csv.',
    ra='(df_goc 4,424 dòng, df_nhi_phan 3,630 dòng, thong_ke).',
    cach=['Đọc CSV.',
          'Chuẩn hóa tên cột: khoảng trắng -> "_", bỏ dấu nháy, ngoặc; "/" -> "_" (ví dụ "Mother\'s qualification" -> "Mothers_qualification").',
          'Giữ dòng có Target là Dropout hoặc Graduate, tức là loại 794 Enrolled (right-censoring: chưa có kết cục).',
          'Tạo Target_Graduate = 1 nếu Graduate, 0 nếu Dropout.',
          'Đếm số tốt nghiệp, bỏ học, nhận học bổng và đóng gói vào ThongKeTapDuLieu.'],
    hoi=[('Vì sao loại Enrolled?', 'Họ chưa có kết cục lúc chốt dữ liệu; gán 0 hay 1 đều là đoán. Mục 9.D kiểm tra lại trên cả 4,424 SV bằng kết cục giữ chân: +16.87%.')])

A['in_thong_tin_moi_truong'] = dict(
    muc='Ghi lại phiên bản Python, thư viện, CPU và revision git của lần chạy (yêu cầu tái lập, Slide 41, 76).',
    cach=['Lấy phiên bản Python, hệ điều hành, CPU.',
          'Import từng gói và đọc __version__; gói nào thiếu thì ghi "không có".',
          'Gọi git rev-parse để lấy mã commit; nếu có thay đổi chưa commit thì ghi thêm chú thích.'],
    luu_y='Giúp người khác dựng lại đúng môi trường; không ảnh hưởng kết quả.')

A['luong_loai_tru'] = dict(
    muc='Bảng luồng loại trừ (eligibility flow, Slide 55): 4,424 -> loại 794 Enrolled -> 3,630, kèm kiểm tra chất lượng.',
    cach=['Đếm số dòng gốc và sau khi lọc Dropout/Graduate.',
          'Đếm dòng trùng lặp hoàn toàn (duplicated trên mọi cột) -> 0.',
          'Đếm giá trị thiếu ở 7 cột dùng trong mô hình -> 0.',
          'Kiểm tra điểm đầu vào nằm trong [95, 200] -> 0 dòng ngoài miền.',
          'Tính tỷ lệ tốt nghiệp 60.85%.'],
    hoi=[('Có xử lý missing/outlier không?', 'Không cần: 0 dòng trùng, 0 giá trị thiếu, 0 điểm ngoài miền hợp lệ; chỉ loại Enrolled.')])

A['kiem_toan_thoi_diem_bien'] = dict(
    muc='Bảng 1.6a: phân loại mọi cột theo thời điểm đo so với học bổng: biến nền (được làm Z) và biến sau nhập học (cấm điều chỉnh).',
    cach=['Lấy mọi cột trừ Target.',
          'Cột bắt đầu bằng "Curricular_units" (tín chỉ, điểm HK1/HK2) là biến SAU nhập học: 12 biến.',
          'Các cột còn lại là biến nền: 24 biến.',
          'Ghi rõ cột nào đang được dùng làm T/Z1..Z5 và quy tắc cho từng nhóm.'],
    ly_thuyet='Điều chỉnh theo biến hậu can thiệp (post-treatment) sẽ chặn mất một phần hiệu ứng đi qua nó, làm ATE bị chệch.',
    hoi=[('Sao không dùng điểm HK1/HK2 làm biến điều chỉnh, dù chúng dự báo rất tốt?', 'Chúng xảy ra sau học bổng, là trung gian T -> M -> Y. Điều chỉnh theo chúng sẽ "ăn mất" hiệu ứng.')])

A['ve_phan_bo_diem_vao'] = dict(
    muc='Hình 1.1: phân bố điểm đầu vào với hai ngưỡng 120/140 của Z1, và tỷ lệ tốt nghiệp trong từng nhóm.',
    cach=['Chia điểm vào thành 3 nhóm bằng pd.cut: <120, [120,140), >=140 (right=False nghĩa là khoảng đóng bên trái).',
          'Trái: histogram 30 cột, kẻ hai vạch đỏ tại 120 và 140.',
          'Phải: cột tỷ lệ tốt nghiệp theo nhóm, ghi n trên mỗi cột.',
          'Trả bảng số SV, tỷ lệ học bổng, tỷ lệ tốt nghiệp theo nhóm.'],
    hoi=[('Ngưỡng 120/140 chọn thế nào?', 'Chốt trước theo ý nghĩa thực tế, không dò trên dữ liệu; hình kiểm tra lại là 3 nhóm đều đủ mẫu (1,075 / 1,869 / 686).')])

A['ve_mutual_information'] = dict(
    muc='Hình 1.2: xếp hạng biến nền theo Mutual Information với Y; tô xanh biến đang dùng, đỏ biến học phí, xám biến không dùng.',
    cach=['Bỏ Target và mọi biến "Curricular_units" (sau nhập học).',
          'Ép các cột về số, ô lỗi thay bằng 0.',
          'mutual_info_classif ước lượng MI (k-NN, random_state=42).',
          'Lấy 15 biến cao nhất và vẽ thanh ngang theo màu.'],
    ly_thuyet='I(X;Y) = sum p(x,y) log[p(x,y) / (p(x)p(y))]; bằng 0 khi độc lập. MI đo LIÊN KẾT, không đo nhân quả.',
    luu_y='mutual_info_classif coi mọi cột là liên tục (ước lượng k-NN), kể cả cột định danh như Course, nên chỉ dùng để xếp hạng tương đối.',
    hoi=[('Có chọn Z theo MI không?', 'Không. Z chọn theo tiêu chuẩn cửa sau (thời điểm + ảnh hưởng cả T và Y); MI chỉ để kiểm tra lại.')])

A['bang_overlap_theo_tang'] = dict(
    muc='Bảng 1.6b: đếm số SV có/không học bổng trong 6 tầng (3 nhóm điểm vào x 2 nhóm tuổi) để kiểm tra overlap thô.',
    cach=['Tạo nhóm điểm (3 mức) và nhóm tuổi (>20 / <=20).',
          'Đếm tổng số SV và số có học bổng trong từng tầng; suy ra số không học bổng và tỷ lệ.',
          'Gắn cờ "tầng thưa" nếu một trong hai nhóm có < 30 SV.'],
    ly_thuyet='Positivity: 0 < P(T=1 | Z=z) < 1 với mọi z. Thiếu một nhóm thì mô hình phải ngoại suy.',
    hoi=[('Overlap có đạt không?', 'Ở mức 6 tầng thì đạt (tầng nhỏ nhất vẫn có 44 SV có học bổng). Khi chia đủ 5 biến thì có vài hồ sơ thiếu hỗ trợ, ghi ở cột Overlap_Status của Table B.')])

A['tinh_toan_do_chenh_lech_quan_sat_tho'] = dict(
    muc='Tính chênh lệch quan sát thô: P(Y=1|T=1) - P(Y=1|T=0) = 86.17% - 51.63% = +34.54 điểm %.',
    cach=['Lấy cột Target_Graduate của nhóm T=1 và T=0.', 'Lấy trung bình mỗi nhóm (= tỷ lệ tốt nghiệp).', 'Trừ hai tỷ lệ.'],
    ly_thuyet='Đây là P(Y|T) (quan sát), KHÔNG phải P(Y|do(T)). Nó trộn hiệu ứng thật với nhiễu nền (SV giỏi dễ có học bổng và dễ tốt nghiệp).',
    hoi=[('Vậy cấp học bổng tăng tốt nghiệp 35% à?', 'Không. Sau điều chỉnh nhiễu, ATE khoảng +24%; khoảng 10 điểm % là thiên lệch do nhiễu quan sát được.')])

A['ve_ty_le_tot_nghiep_theo_nhom'] = dict(
    muc='Hình 2.1: tỷ lệ tốt nghiệp theo học bổng (trái) và theo học bổng x học phí đúng hạn (phải), tô đỏ ô thưa n < 100.',
    cach=['groupby Scholarship_holder: trung bình và số lượng -> cột trái.',
          'groupby (Scholarship_holder, Tuition_fees_up_to_date) -> 4 cột phải; nhãn sinh tự động từ chỉ số groupby nên không lệch.',
          'Cột có n < 100 tô đỏ (ô T=1, chưa đóng đúng hạn chỉ có 37 SV).'],
    hoi=[('24.3% ở cột đỏ có tin được không?', 'Là 9/37; CI 95% Wilson khoảng [13%, 40%]. Ô này cho thấy positivity yếu, không dùng làm ước lượng chính xác.')])

A['ve_do_thi_co_huong'] = dict(
    muc='Hàm vẽ đồ thị có hướng dùng chung cho Hình 3.1, 6.1, 8.1: vị trí nút cố định, tô đỏ / nét đứt các cạnh được chỉ định.',
    vao='ax, danh sách cạnh, vị trí nút, nhãn nút, tiêu đề, tập cạnh đỏ, tập cạnh nét đứt, độ cong (số hoặc dict theo cạnh), kích thước nút.',
    cach=['Tạo nx.DiGraph với đủ nút và cạnh.', 'Vẽ nút và nhãn.',
          'Vẽ từng cạnh riêng: đỏ nếu thuộc canh_do, nét đứt nếu thuộc canh_net_dut; độ cong lấy từ do_cong (dict thì tra theo cạnh, mặc định 0).'],
    luu_y='Tham số do_cong dạng dict được thêm để Hình 8.1 vẽ cạnh Z -> Z vòng ra ngoài, không đè lên nút khác.')

A['vi_tri_theo_tang'] = dict(
    muc='Tính tọa độ nút cho Hình 6.1 theo tầng thời gian: Z cột trái, T và M1 hạ xuống hàng dưới, Y bên phải.',
    cach=['Trục x = số tầng x 2.2.', 'Các nút cùng tầng xếp dọc cách đều quanh y = 0.',
          'Tầng thuộc hang_duoi (T, M1) đặt ở y = -do_ha để cạnh Z -> Y không chạy xuyên qua T.'],
    luu_y='Chỉ phục vụ hiển thị, không ảnh hưởng tính toán.')

A['do_cong_canh_z_z'] = dict(
    muc='Gán độ cong cho cạnh nối hai biến Z cùng cột ở Hình 8.1: cạnh kề nhau vẽ thẳng, cạnh cách xa vòng ra bên trái.',
    cach=['Chỉ xét cạnh có cả hai đầu là Z.', 'Số bước = khoảng cách dọc / 2; cạnh kề nhau (1 bước) để thẳng.',
          'Cạnh xa: rad = (muc_cong + 0.7*(số bước - 2)) / |dy|, đổi dấu theo chiều đi lên hay xuống để luôn vòng sang trái.'],
    luu_y='Chỉ phục vụ hiển thị.')

A['ve_forest_ate_table_b'] = dict(
    muc='Hình 8.3: forest plot ATE và CI 95% theo từng seed cho B0, H1, H2, kèm vạch chênh lệch thô.',
    cach=['Lấy các dòng theo seed (bỏ dòng tổng hợp), ép ATE/CI về số.',
          'Mỗi dòng một điểm với thanh sai số (CI bootstrap); khoảng âm được chặn ở 0 để tránh lỗi khi CI suy biến.',
          'Kẻ vạch xám tại chênh lệch thô +0.3454.'],
    hoi=[('Hình nói lên điều gì?', 'Cả ba mô hình đều quanh +0.24, cách xa +0.345 thô: kết luận can thiệp không phụ thuộc mô hình.')])

A['hoc_cau_truc_hill_climb_bic'] = dict(
    muc='Minh họa (mục 3) HillClimb + BIC trên 4 biến, có hoặc không có ExpertKnowledge cấm cạnh.',
    vao='du_lieu (4 biến rời rạc), danh_sach_cam (None = học tự do), so_vong_lap_toi_da.',
    ra='Danh sách cạnh học được.',
    cach=['Tạo HillClimbSearch và hàm điểm BIC.', 'Nếu có danh sách cấm thì gói vào ExpertKnowledge(forbidden_edges=...).',
          'estimate(): leo đồi bằng thêm/xóa/đảo cạnh, giữ thay đổi làm điểm tăng nhiều nhất.'],
    ly_thuyet='BIC = log-likelihood - (số tham số / 2) x log N: phạt đồ thị phức tạp.',
    hoi=[('Kết quả minh họa là gì?', 'Học tự do ra cạnh ngược Y -> Z; cấm cạnh ra từ Y thì hết nhưng còn T -> Debtor vì chưa cấm T -> Z. Bài học: ràng buộc phải đủ Z -> T -> Y, như ở B0 mục 8.B.')])

A['so_sanh_hoc_tham_so_mle_va_bayesian'] = dict(
    muc='So sánh CPD của Y ước lượng bằng MLE (đếm tần suất) và bằng Bayes với tiên nghiệm BDeu (ESS = 10).',
    cach=['Dựng mạng Bayes con với cạnh Học bổng -> Y, Debtor -> Y.',
          'MLE: P(y | cha) = N(y, cha) / N(cha).',
          'BDeu: P(y | cha) = (N(y, cha) + ESS/(r q)) / (N(cha) + ESS/q), r = số trạng thái của Y, q = số cấu hình cha.'],
    ly_thuyet='BDeu là tiên nghiệm Dirichlet đều, tương đương thêm ESS "mẫu ảo" chia đều cho các ô; ô càng thưa càng bị kéo về đều.',
    hoi=[('Vì sao dùng BDeu thay MLE?', 'Ô thưa (ví dụ Debtor=1 và có học bổng) thì MLE dễ ra 0% hoặc 100%; BDeu làm mịn, tránh xác suất cực đoan.')])

A['KetQuaKiemTraRangBuoc'] = dict(muc='Dataclass kết quả của một bài kiểm tra: có hợp lệ không, tên bài kiểm tra, danh sách vi phạm.')

A['kiem_tra_tinh_khong_chu_trinh'] = dict(
    muc='Kiểm tra đồ thị có phải DAG (không có chu trình) không; nếu không thì liệt kê chu trình.',
    cach=['nx.is_directed_acyclic_graph.', 'Nếu có chu trình, liệt kê bằng nx.simple_cycles.'],
    ly_thuyet='Mạng Bayes chỉ phân rã P(X) = prod P(X_i | Pa(X_i)) được khi đồ thị không có chu trình.')

A['kiem_tra_phan_tang_thoi_gian'] = dict(
    muc='Kiểm tra không có cạnh đi ngược thời gian: với mỗi cạnh u -> v, tầng(u) phải <= tầng(v).',
    cach=['Duyệt mọi cạnh, tra tầng của hai đầu trong bang_phan_tang (Z=1, T=2, M1=3, Y=4).', 'Cạnh có tầng nguồn lớn hơn tầng đích là vi phạm.'],
    hoi=[('Cạnh cùng tầng có bị cấm không?', 'Không (điều kiện là >, không phải >=); chiều giữa các Z cùng tầng để dữ liệu/chuyên gia quyết.')])

A['kiem_tra_danh_sach_chuyen_gia'] = dict(
    muc='Kiểm tra blacklist (cạnh cấm không được có) và whitelist (cạnh bắt buộc phải có).',
    cach=['Mỗi cạnh trong danh_sach_cam mà có mặt -> vi phạm.', 'Mỗi cạnh trong danh_sach_bat_buoc mà vắng mặt -> vi phạm.'])

A['kiem_tra_d_separation_backdoor'] = dict(
    muc='Kiểm tra tập Z có thỏa tiêu chuẩn cửa sau không: chặn mọi đường cửa sau giữa T và Y.',
    cach=['Tạo đồ thị G gạch dưới T: xóa mọi cạnh ĐI RA từ T (giữ lại đường đi VÀO T, tức đường cửa sau).',
          'Dùng pgmpy DAG.is_dconnected(T, Y, observed=Z): nếu không còn d-connected thì Z chặn hết cửa sau.'],
    ly_thuyet='Tiêu chuẩn cửa sau (Pearl): Z không chứa con cháu của T và T d-tách Y khi biết Z trong G gạch dưới T. Khi đó P(y|do(t)) = sum_z P(y|t,z) P(z).',
    hoi=[('Vì sao xóa cạnh đi ra từ T?', 'Để chỉ còn lại các đường không nhân quả (cửa sau); đường nhân quả T -> ... -> Y không được chặn.')])

A['KetQuaUocLuongDoWhy'] = dict(muc='Dataclass kết quả ước lượng DoWhy: tên phương pháp, ATE, tập biến cửa sau.')
A['KetQuaKiemDinhRefute'] = dict(muc='Dataclass kết quả một refuter: tên, hiệu ứng gốc, hiệu ứng sau bác bỏ, kết luận.')

A['thuc_thi_quy_trinh_can_thiep_dowhy'] = dict(
    muc='Chạy đủ 4 bước DoWhy trên Domain DAG (mục 7): model -> identify -> estimate (2 cách) -> refute (3 cách).',
    vao='Dữ liệu 3,630 SV (biến gốc), tên T, tên Y, đồ thị dạng chuỗi DOT.',
    ra='(model, kết quả hồi quy tuyến tính, kết quả propensity score weighting, danh sách 3 refuter).',
    cach=['CausalModel(data, treatment, outcome, graph).',
          'identify_effect: DoWhy tự tìm tập cửa sau (5 biến Z).',
          'estimate_effect "backdoor.linear_regression": hồi quy Y ~ T + Z, hệ số của T là ATE (+24.12%).',
          'estimate_effect "backdoor.propensity_score_weighting": trọng số nghịch đảo xác suất nhận học bổng (+25.16%).',
          'Refuter 1 random_common_cause: thêm một biến nhiễu ngẫu nhiên -> ATE không được đổi (lệch < 0.02).',
          'Refuter 2 placebo_treatment_refuter: thay T bằng biến giả -> ATE phải về gần 0 (< 0.03).',
          'Refuter 3 data_subset_refuter 80%: ước lượng lại trên 80% dữ liệu -> ATE ổn định (lệch < 0.02).'],
    luu_y='np.random.seed(42) trước mỗi refuter vì DoWhy vẫn dùng bộ sinh số toàn cục của numpy; nhờ vậy số tái lập được.',
    hoi=[('Refuter qua thì chứng minh không có nhiễu ẩn à?', 'Không. Refuter chỉ kiểm tra độ ổn định của bộ ước lượng; nhiễu ẩn được đánh giá bằng RV ở mục 9.H.')])

A['Config'] = dict(
    muc='Dataclass "đóng băng" chứa MỌI cấu hình của mục 8: seed, tỷ lệ split, tên nút, thư mục, siêu tham số B0/H1/H2, tham số đánh giá.',
    cach=['seeds 42..46 cho 5 run ghép cặp; split 70/15/15.',
          'B0: ESS=5, lưới hàm điểm {BIC, BDeu, K2}, cạnh bắt buộc T -> Y.',
          'H1: 32 nơ-ron ẩn, dropout 0.2, lr 0.005, weight decay 1e-4, batch 64, tối đa 150 epoch, patience 15.',
          'H2: 10 nơ-ron ẩn/nút, lưới lambda_L1 {1e-3, 1e-4}, lưới tau {0.1, 0.2, 0.3, 0.5, 0.8}, lr 0.01, 300 bước trong, tối đa 12 vòng ngoài, h_tol 1e-6.',
          'Đánh giá: 1000 bootstrap, 30 lần mô phỏng refuter, subset 80%.'],
    luu_y='frozen=True: không sửa được sau khi tạo, tránh vô tình đổi cấu hình giữa chừng.',
    hoi=[('Siêu tham số chọn thế nào?', 'Hàm điểm B0, tau và lambda của H2, ngưỡng F1 đều chọn trên VALIDATION; phần còn lại cố định trước, không dò theo TEST.')])

A['init_directories'] = dict(muc='Tạo các thư mục artifacts/ (splits, tables, checkpoints, graphs) nếu chưa có.')

A['set_seed'] = dict(
    muc='Cố định mọi nguồn ngẫu nhiên: random, numpy, torch (và CUDA nếu có), PYTHONHASHSEED.',
    luu_y='Gọi ở đầu mỗi lần huấn luyện để cùng seed thì ra cùng kết quả.')

A['load_and_preprocess_uci697'] = dict(
    muc='Nạp UCI 697 và mã hóa đúng 7 biến của mô hình chính: Z1..Z5, T, Y (mã hóa dùng cho Tables A–C).',
    cach=['Đọc data/uci_697.csv (nếu thiếu thì tải từ UCI và lưu cache).',
          'Chuẩn hóa tên cột bằng cách bỏ ký tự đặc biệt rồi tra bảng alias (xử lý BOM, dấu nháy).',
          'Loại Enrolled: 4,424 -> 3,630.',
          'Z1 = điểm vào 3 mức (<120 -> 0, [120,140) -> 1, >=140 -> 2).',
          'Z2 = Debtor, Z3 = học phí đúng hạn (0/1).',
          'Z4 = học vấn mẹ >= 13 (đại học trở lên) -> 1.',
          'Z5 = tuổi nhập học > 20 -> 1.',
          'T = Scholarship holder; Y = Graduate 1 / Dropout 0.',
          'assert không có NaN.'],
    luu_y='Bẫy ký hiệu: ở mục 8, Z1 = điểm vào, Z2 = Debtor, Z3 = học phí, Z4 = học vấn mẹ, Z5 = tuổi. Mục 6–7 và 9.H đánh số khác.')

A['get_or_create_splits'] = dict(
    muc='Tạo (hoặc nạp lại) 5 phân chia TRAIN/VAL/TEST 70/15/15 cho 5 seed, phân tầng theo tổ hợp (Y, T), lưu JSON.',
    cach=['Tầng phân chia = chuỗi "Y_T" để tỷ lệ tốt nghiệp và học bổng giống nhau ở mọi tập.',
          'Nếu file split đã có thì đọc lại (đóng băng).',
          'Nếu chưa: train_test_split tách 70% / 30%, rồi tách 30% thành 15% VAL / 15% TEST (đều stratify).',
          'Lưu chỉ số ra artifacts/splits/split_seed_<s>.json.',
          'assert 3 tập không giao nhau và phủ đủ 3,630 dòng.'],
    ly_thuyet='Slide 56: dùng chung TRAIN/VAL/TEST cho mọi mô hình; C_test (545 hồ sơ) là quần thể tham chiếu để chuẩn hóa can thiệp.',
    hoi=[('Vì sao 5 seed?', 'Slide 39 yêu cầu ít nhất 5 run ghép cặp; B0/H1/H2 dùng cùng split trong mỗi seed nên so sánh được theo cặp.')])

A['build_b0_dag'] = dict(
    muc='Dựng Domain DAG tham chiếu 7 nút, 11 cạnh (mỗi Z -> T, mỗi Z -> Y, T -> Y).',
    luu_y='KHÔNG dùng để khớp B0. B0 học DAG từ dữ liệu; đồ thị này chỉ để so sánh.')

A['temporal_forbidden_edges'] = dict(
    muc='Danh sách cạnh cấm theo thời gian, dùng chung cho B0 (ExpertKnowledge) và H2 (mặt nạ M).',
    cach=['Cấm mọi cạnh đi ra từ Y: Y -> T, Y -> Z1..Z5 (kết cục không gây ra quá khứ).',
          'Cấm mọi cạnh T -> Zk (học bổng không gây ra biến nền).'],
    luu_y='Cạnh giữa các Z với nhau KHÔNG bị cấm: chiều giữa các biến nền để thuật toán tự học.',
    hoi=[('Có cấm Z -> Z không?', 'Không. Các Z cùng tầng; chiều giữa chúng không ảnh hưởng truy vấn do(T) vì đều là tiền can thiệp.')])

A['fit_discrete_bayesian_model'] = dict(
    muc='Khớp mọi CPD của một mạng Bayes rời rạc bằng ước lượng Bayes với tiên nghiệm BDeu (ESS mặc định 5).',
    cach=['Thử API mới của pgmpy (DiscreteBayesianEstimator); nếu lỗi thì dùng API cũ (BayesianEstimator).', 'model.check_model() để chắc mỗi CPD hợp lệ (tổng xác suất = 1, khớp cha).'],
    ly_thuyet='Mỗi CPD là P(X | Pa(X)) = (đếm + ESS/(r q)) / (đếm cha + ESS/q): đếm tần suất có làm mịn. KHÔNG có epoch hay gradient.')

A['fit_bn_on_dag'] = dict(
    muc='Từ một DAG (networkx) tạo DiscreteBayesianNetwork đủ 7 nút rồi khớp CPD BDeu trên TRAIN.',
    luu_y='Dùng cho B0 và cả H2: hai mô hình chỉ khác ĐỒ THỊ, cùng họ CPD và cùng ESS, nên so sánh công bằng.')

A['validation_log_likelihood'] = dict(
    muc='Log-likelihood của toàn mạng trên một tập dữ liệu: sum_i sum_v log P(x_v | pa(x_v)). Dùng để CHỌN trên VALIDATION.',
    cach=['Với mỗi nút, lấy CPD.', 'Đổi giá trị của nút và các cha thành chỉ số trong bảng CPD (searchsorted theo state_names).',
          'Tra xác suất của từng dòng, lấy log, cộng dồn.'],
    hoi=[('Vì sao chọn hàm điểm bằng log-likelihood VAL chứ không phải bằng chính hàm điểm?', 'Ba hàm điểm có thang khác nhau nên không so trực tiếp được; log-likelihood trên dữ liệu chưa thấy là thước đo chung, khách quan.')])

A['learn_b0_dag'] = dict(
    muc='Tìm sơ đồ mũi tên (DAG) cho B0 từ dữ liệu TRAIN. Vì có nhiều cách "chấm điểm" sơ đồ cho ra kết quả khác nhau, hàm thử 3 cách rồi để dữ liệu chưa thấy (VALIDATION) quyết định cách nào đúng hơn.',
    vao='train_df (2,541 SV), val_df (544 SV), cfg.',
    ra='(DAG được chọn, tên hàm điểm thắng, bảng 3 ứng viên để báo cáo).',
    cach=['Khai báo luật bắt buộc trước khi học (ExpertKnowledge): CẤM mọi mũi tên đi ngược thời gian (Y -> bất kỳ, T -> Z) và BẮT BUỘC có T -> Y. '
          'Vì sao: dữ liệu quan sát chỉ cho biết hai biến "đi cùng nhau", không cho biết chiều; không cấm thì thuật toán có thể vẽ "tốt nghiệp gây ra điểm đầu vào" (đã thấy ở mục 3).',
          'Với từng "giám khảo" BIC, BDeu (ESS=5), K2: chạy HillClimb — bắt đầu từ sơ đồ trống, thử thêm / bớt / đảo một mũi tên, giữ thay đổi làm điểm tăng nhiều nhất, lặp đến khi không cải thiện. '
          'Kết quả: 3 sơ đồ, khác nhau chủ yếu ở số mũi tên.',
          'Vì sao 3 giám khảo cho 3 sơ đồ khác nhau: chúng khác nhau ở mức "khó tính" với mũi tên mới. BIC phạt mỗi tham số thêm rất nặng (theo log N); BDeu với ESS nhỏ cũng khá khắt khe; '
          'K2 dễ dãi hơn nên giữ thêm mũi tên. Không có lý thuyết nào nói trước giám khảo nào đúng cho dữ liệu này.',
          'Để dữ liệu tự phân xử: với mỗi sơ đồ, điền bảng CPD trên TRAIN rồi tính log-likelihood trên VALIDATION — tức "sơ đồ này đoán dữ liệu CHƯA TỪNG THẤY tốt cỡ nào". '
          'Không so trực tiếp điểm BIC với điểm K2 được vì chúng ở hai thang đo khác nhau; log-likelihood trên VAL là thước đo chung, khách quan.',
          'Chọn sơ đồ có log-likelihood VAL cao nhất (ít âm nhất), rồi đóng băng. Kết quả: K2 thắng ở cả 5 seed.',
          'assert: sơ đồ không có vòng tròn và không chứa mũi tên bị cấm — lưới an toàn, nếu sai thì dừng ngay chứ không chạy tiếp với sơ đồ hỏng.'],
    ly_thuyet='Số liệu thật cho thấy vì sao K2 thắng: ở seed 42, BIC và BDeu cùng ra 9 mũi tên với Pa(Y) = {T, học phí, tuổi}, log-likelihood VAL = -2211.11; '
              'K2 ra 11 mũi tên, giữ thêm Debtor -> Y, log-likelihood VAL = -2208.37 (tốt hơn 2.74). Quy luật lặp lại ở cả 5 seed: giám khảo khắt khe bỏ Debtor -> Y, '
              'K2 giữ lại, và lần nào sơ đồ có Debtor -> Y cũng đoán VAL tốt hơn (seed 46: -2174.84 so với -2182.63; seed 44 BDeu cũng giữ Debtor và suýt bằng K2). '
              '=> Debtor thực sự giúp dự đoán tốt nghiệp trên dữ liệu mới; BIC đã "cắt quá tay". Đây là lý do chọn bằng VAL thay vì chọn cố định một giám khảo từ đầu.',
    luu_y='T -> Y bắt buộc KHÔNG ép học bổng phải có tác dụng: nó chỉ bảo đảm sơ đồ có chỗ để đo. Độ lớn tác dụng vẫn do bảng CPD học từ dữ liệu quyết định; nếu học bổng vô dụng, '
          'hai cột "có / không học bổng" trong bảng sẽ gần bằng nhau và ATE ≈ 0. Bằng chứng: H2 không bị ép mà vẫn tự học ra T -> Y ở cả 5 seed.',
    hoi=[('Vì sao không dùng luôn BIC cho đơn giản?', 'Vì BIC bỏ mất Debtor -> Y, và trên dữ liệu chưa thấy (VAL) sơ đồ đó đoán kém hơn ở cả 5 seed. Chọn bằng VAL là để dữ liệu quyết định, không phải nhóm chọn theo ý thích.'),
         ('Chọn trên VAL có làm rò rỉ TEST không?', 'Không. Chỉ dùng TRAIN để học và VAL để chọn; TEST chỉ dùng một lần ở cuối cho Table A, B.'),
         ('B0 có "train" không?', 'Có, nhưng không bằng gradient: tìm sơ đồ bằng leo đồi, điền bảng bằng đếm có làm mịn BDeu; lựa chọn duy nhất là giám khảo, chọn trên VAL.'),
         ('Sao phải ép T -> Y? Như vậy là áp đặt kết quả à?', 'Không áp đặt độ lớn: cạnh bắt buộc chỉ để câu hỏi do(T) có nghĩa, còn tác dụng lớn hay nhỏ vẫn do dữ liệu. H2 không bị ép mà vẫn tự học ra cạnh này.')])

A['save_graph_artifacts'] = dict(
    muc='Lưu danh sách nút, cạnh và thông tin phụ (hàm điểm, tau, log H2...) ra artifacts/graphs/<tên>.json.',
    luu_y='File JSON này là nơi notebook xem log (training_log_viewer.ipynb) đọc lại log H2.')

A['query_model_effects'] = dict(
    muc='Hàm truy vấn CHUNG cho B0/H1/H2: với mỗi hồ sơ C_i tính q(0, C_i), q(1, C_i) và xác suất dự báo P(Y=1 | T thật, Pa(Y)).',
    vao='model (mạng Bayes đã khớp), df (các hồ sơ, ví dụ C_test), cfg.',
    ra='(q0, q1, pred) — mỗi cái là mảng dài bằng số hồ sơ.',
    cach=['Lấy CPD của Y; cha của Y là Pa(Y).',
          'Nếu Y không có cha: mọi hồ sơ cùng một xác suất.',
          'Tạo chỉ số tra bảng: chiều đầu = 1 (lấy P(Y=1)); với cha là T thì đặt 0 (cho q0), 1 (cho q1), hoặc T thật (cho pred); cha khác lấy giá trị của hồ sơ.',
          'Tra bảng CPD bằng numpy advanced indexing cho cả 545 hồ sơ cùng lúc (khoảng 0.2 ms).'],
    ly_thuyet='q_m(t, C_i) = P_m(Y=1 | T=t, Pa(Y)\\T = C_i). Vì mặt nạ cấm T -> Z, mọi cha khác của Y là tiền can thiệp nên là tập điều chỉnh cửa sau hợp lệ, do đó q_m(t, C_i) = P(Y=1 | do(T=t), C_i). '
              'risk_m(t) = trung bình q_m(t, C_i) trên C_test; ATE = risk(1) - risk(0). Không dùng Y_test.',
    hoi=[('Sao không dùng Variable Elimination của pgmpy?', 'Kết quả như nhau vì chỉ cần CPD của Y; tra bảng trực tiếp nhanh và minh bạch hơn (mục 8.2c tính tay lại để đối chiếu).')])

A['NeuralCPDModule'] = dict(
    muc='Mạng MLP ước lượng P(Y=1 | Pa(Y)) cho H1. Đầu ra là logit (chưa qua sigmoid).',
    cach=['Linear(số chiều vào -> 32) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 1).',
          'forward trả logit; xác suất = sigmoid(logit).'],
    ly_thuyet='Với Pa(Y) = {T, Z2, Z3, Z5} (4 biến nhị phân) thì số tham số = (4x32+32) + 64 (BatchNorm) + (32x16+16) + (16+1) = 769.',
    hoi=[('Vì sao cần BatchNorm và Dropout?', 'BatchNorm ổn định huấn luyện; Dropout và weight decay chống overfit. Thực tế đầu vào chỉ 16 tổ hợp nên mạng hội tụ rất nhanh.')])

A['encode_h1_features'] = dict(
    muc='Mã hóa ĐÚNG các nút cha của Y làm đầu vào MLP (Slide 60): biến 3 trạng thái -> one-hot 3 chiều, biến nhị phân giữ nguyên.',
    luu_y='STATE_CARD là bảng số trạng thái của từng nút (Z1 có 3, còn lại 2).')

A['h1_input_dim'] = dict(muc='Số chiều đầu vào của MLP: cộng 3 cho mỗi biến 3 trạng thái, 1 cho mỗi biến nhị phân.')

A['train_h1_mlp'] = dict(
    muc='Huấn luyện MLP của H1 trên TRAIN bằng BCE, theo dõi loss VAL sau mỗi epoch; mặc định có early stopping.',
    vao='train_df, val_df, parents = Pa(Y) từ DAG B0, cfg, seed, dung_som (True mặc định).',
    ra='(model, nhat_ky) — nhật ký gồm train_loss, val_loss từng epoch, best_epoch, epochs_run, số tham số, kiến trúc.',
    cach=['set_seed; mã hóa X_train, X_val; DataLoader batch 64, xáo trộn.',
          'Loss = BCEWithLogitsLoss (sigmoid + binary cross-entropy, ổn định số học); Adam lr 0.005, weight decay 1e-4.',
          'Mỗi epoch: duyệt các batch, lan truyền ngược, cập nhật; cộng dồn loss TRAIN (dropout đang bật).',
          'Cuối epoch: model.eval(), tính loss VAL (không dropout).',
          'Nếu loss VAL tốt hơn kỷ lục: lưu bản sao trọng số, đặt lại bộ đếm; ngược lại tăng bộ đếm.',
          'dung_som=True: bộ đếm đạt 15 thì dừng và khôi phục trọng số tốt nhất. dung_som=False: chạy đủ 150 epoch, giữ trọng số epoch cuối.'],
    hoi=[('Bao nhiêu epoch?', 'Tối đa 150; thực tế dừng sau 17–39 epoch, tốt nhất ở epoch 2–4 (seed 43 là 24) vì đầu vào chỉ có 16 tổ hợp.'),
         ('Loss TRAIN cao hơn VAL là sao?', 'Loss TRAIN đo khi dropout bật nên bị đội lên; loss VAL đo ở chế độ eval.')])

A['save_h1_checkpoint'] = dict(
    muc='Lưu trọng số và nhật ký H1 ra artifacts/checkpoints/h1_seed<seed><hậu tố>.pt.',
    luu_y='Hậu tố "_early_stop" / "_no_early_stop" dùng ở mục 8.E2 để lưu hai bản riêng.')

A['build_h1_bayesian_network'] = dict(
    muc='Biến MLP thành bảng CPD của Y rồi gắn vào bản sao của B0: H1 = B0 với CPD(Y) do mạng nơ-ron sinh ra.',
    cach=['Liệt kê MỌI cấu hình cha của Y (itertools.product các trạng thái).',
          'Cho MLP dự đoán P(Y=1) ở từng cấu hình; chặn trong [1e-4, 1 - 1e-4] để tránh log 0.',
          'Tạo TabularCPD với 2 hàng [1-p, p].',
          'Dựng mạng mới cùng cạnh với B0, chép mọi CPD khác của B0, thay CPD của Y bằng CPD mới; check_model.'],
    ly_thuyet='Vì H1 vẫn là một mạng Bayes trên cùng DAG, truy vấn can thiệp dùng y hệt query_model_effects như B0: so sánh công bằng, chỉ khác cách tham số hóa CPD(Y).',
    hoi=[('H1 khác B0 ở đâu?', 'Cùng DAG, cùng CPD các nút khác; chỉ CPD của Y được học bằng MLP thay vì đếm BDeu.')])

A['build_domain_mask'] = dict(
    muc='Tạo mặt nạ M (7x7) cho H2: M[i, j] = 0 nếu cạnh i -> j bị cấm (cùng danh sách với B0) và cấm tự nối.',
    ly_thuyet='Slide 63: H2 phải dùng cùng ràng buộc miền với B0. Nhân M vào trọng số làm cạnh cấm luôn bằng 0.')

A['NotearsMLP'] = dict(
    muc='Mạng NOTEARS phi tuyến: mỗi nút j có một MLP riêng f_j dự đoán X_j từ các nút khác; độ mạnh cạnh i -> j suy ra từ trọng số tầng 1.',
    cach=['fc1 có kích thước [nút j, nơ-ron ẩn, đầu vào i]; masked_fc1 nhân mặt nạ M nên đầu vào bị cấm có trọng số và gradient bằng 0.',
          'forward: h = sigmoid(fc1 . X + b1) cho từng nút, rồi X_hat_j = fc2 . h + b2.',
          'adjacency_sq: W[i, j]^2 = tổng bình phương trọng số tầng 1 từ đầu vào i vào MLP của nút j.'],
    ly_thuyet='Zheng et al. (2020, NOTEARS-MLP): W[i, j] = chuẩn L2 của cột trọng số tầng 1; nếu W[i, j] = 0 thì f_j không phụ thuộc X_i, tức không có cạnh i -> j.')

A['train_notears_mlp'] = dict(
    muc='Tối ưu NOTEARS-MLP bằng augmented Lagrangian để vừa khớp dữ liệu vừa ép đồ thị không có chu trình.',
    vao='train_df (7 biến), cfg, seed, lambda_l1.',
    ra='Dict: W, mask, h_final, converged, outer_iters, h_history, recon_history, forbidden_weight_max, lambda_l1.',
    cach=['Chuẩn hóa X (trừ trung bình, chia độ lệch chuẩn).',
          'Hàm mục tiêu mỗi bước: recon + lambda_1 x ||W||_1 + alpha x h + (rho/2) x h^2, với recon = MSE/2 tái tạo X và h = tr(exp(W∘W)) - d.',
          'Vòng TRONG: 300 bước Adam (lr 0.01).',
          'Sau vòng trong: nếu h không giảm xuống dưới 1/4 lần trước thì tăng rho x 10 và chạy lại vòng trong.',
          'Vòng NGOÀI: cập nhật alpha += rho x h; ghi h và recon vào lịch sử; dừng khi h < 1e-6 hoặc hết 12 vòng.',
          'Trả W = căn của adjacency_sq và kiểm tra trọng số lớn nhất ở ô bị cấm (phải bằng 0).'],
    ly_thuyet='h(W) = 0 khi và chỉ khi đồ thị không có chu trình. Augmented Lagrangian biến ràng buộc cứng h = 0 thành chuỗi bài toán có phạt ngày càng nặng.',
    hoi=[('H2 có epoch không?', 'Không. Mỗi vòng ngoài gồm 300 bước Adam; cả 5 seed hội tụ sau 7–8 vòng với h khoảng 1e-7.'),
         ('Vì sao sai số tái tạo tăng qua các vòng?', 'Phạt chu trình càng nặng thì mô hình càng phải bỏ cạnh tạo vòng, chấp nhận khớp kém hơn: đánh đổi giữa khớp dữ liệu và là DAG hợp lệ.')])

A['graph_from_weights'] = dict(
    muc='Chuyển ma trận W thành DAG: giữ cạnh có |W| >= tau, cắt chu trình còn sót, rồi thêm cạnh bắt buộc T -> Y nếu thiếu.',
    cach=['Thêm mọi cạnh i -> j có W[i, j] >= tau.',
          'Trong khi còn chu trình: tìm một chu trình và bỏ cạnh có trọng số nhỏ nhất trong đó (luật sửa khai báo trước, Slide 79).',
          'Nếu thiếu T -> Y thì thêm và bật cờ "forced".'],
    luu_y='Thực tế chưa seed nào phải cắt chu trình hay ép T -> Y ở tau được chọn.')

A['select_h2_graph'] = dict(
    muc='Chọn ngưỡng tau của H2 trên VALIDATION: với mỗi tau, khớp CPD BDeu trên TRAIN và tính log-likelihood VAL, giữ tau tốt nhất.',
    luu_y='Bảng tau_table được lưu vào JSON để xem lại; tau thường là 0.1 hoặc 0.2.')

A['evaluate_prediction_metrics'] = dict(
    muc='Tính chỉ số dự báo trên TEST (Slide 74): AUROC, Macro-F1, Brier, log-loss, precision, recall.',
    cach=['Dò 31 ngưỡng từ 0.2 đến 0.8 trên VALIDATION, chọn ngưỡng có Macro-F1 cao nhất.',
          'Áp ngưỡng đó cho TEST để tính Macro-F1, precision, recall.',
          'AUROC, Brier, log-loss tính trực tiếp trên xác suất (không cần ngưỡng).'],
    hoi=[('Vì sao Macro-F1 của B0 và H1 trùng nhau?', 'Pa(Y) chỉ có 16 cấu hình, hai mô hình xếp hạng giống nhau tại ngưỡng chọn, nên gán nhãn giống hệt; khác biệt chỉ ở xác suất (AUROC, Brier).')])

A['compute_bootstrap_ate_ci'] = dict(
    muc='CI 95% của ATE bằng percentile bootstrap: lấy mẫu lại có hoàn lại các hồ sơ C_test, GIỮ CỐ ĐỊNH mô hình.',
    cach=['Tính hiệu từng hồ sơ d_i = q1_i - q0_i.', 'Lặp 1000 lần: lấy mẫu lại d có hoàn lại, tính trung bình.', 'Lấy phân vị 2.5% và 97.5%.'],
    luu_y='CI này chỉ phản ánh bất định do chọn quần thể tham chiếu; chưa gồm bất định tham số hay đồ thị. Độ dao động giữa 5 seed bổ sung phần đó.',
    hoi=[('CI hẹp (khoảng ±0.008) có nghĩa ATE rất chính xác?', 'Không hẳn: mô hình được giữ cố định. Bất định do học lại mô hình thể hiện qua độ tản giữa 5 seed (Table C).')])

A['identify_with_dowhy'] = dict(
    muc='Chạy DoWhy identify_effect trên CHÍNH DAG của từng mô hình (Slide 57, 62) để xác nhận hiệu ứng định danh được.',
    ra='Dict: status ("identified (backdoor)" hoặc lỗi), backdoor_set, text.',
    luu_y='proceed_when_unidentifiable=False: nếu không định danh được thì báo lỗi thay vì đoán.')

A['overlap_report'] = dict(
    muc='Đếm hồ sơ C_test "không được hỗ trợ": trong TRAIN không có ai cùng cấu hình biến điều chỉnh ở CẢ T=0 và T=1.',
    cach=['groupby (biến điều chỉnh, T) trên TRAIN, đếm số người.', 'Cấu hình được hỗ trợ nếu đếm ở T=0 và T=1 đều > 0.', 'Đếm hồ sơ test có cấu hình không được hỗ trợ.'],
    hoi=[('Hồ sơ không được hỗ trợ thì xử lý sao?', 'Được ghi rõ ở cột Overlap_Status (1–2% ở H2); ước lượng ở các hồ sơ đó dựa vào làm mịn BDeu.')])

A['record_model_run'] = dict(
    muc='Đánh giá một mô hình trong một seed: dự báo trên TEST, can thiệp chuẩn hóa trên C_test, CI, định danh, overlap; ghi vào hai danh sách kết quả.',
    cach=['query_model_effects trên VAL (để chọn ngưỡng F1) và trên TEST (q0, q1, xác suất dự báo).',
          'ATE = trung bình q1 - trung bình q0; CI bootstrap.',
          'evaluate_prediction_metrics: chỉ số dự báo, dùng Y_test.',
          'Tập điều chỉnh = Pa(Y) bỏ T; identify_with_dowhy; overlap_report.',
          'Thêm một dòng vào prediction_results (Table A) và một dòng vào intervention_results (Table B).'],
    ly_thuyet='Y_test CHỈ dùng để chấm dự báo (Table A). ATE (Table B) chỉ dùng hiệp biến C_test, không dùng Y_test (Slide 56).')

A['mean_std'] = dict(muc='Định dạng chuỗi "trung bình ± độ lệch chuẩn" cho dòng tổng hợp 5 seed của Tables A–C.')

A['ve_qua_trinh_huan_luyen'] = dict(
    muc='Hình 8.2: (a) loss TRAIN/VAL của H1 theo epoch, chấm tại epoch tốt nhất; (b) h(W) của H2 theo vòng ngoài (thang log); (c) sai số tái tạo của H2.',
    luu_y='B0 không có đường cong vì không học bằng gradient.')

A['bang_overfit_theo_tap'] = dict(
    muc='Bảng 8.E2: AUROC, accuracy (ngưỡng 0.5) và log-loss của B0/H1/H2 trên TRAIN, VAL, TEST cho cả 5 seed.',
    cach=['Với mỗi seed, mô hình và tập: lấy xác suất dự báo bằng query_model_effects, tính 3 chỉ số.',
          'Tóm tắt trung bình ± độ lệch chuẩn và chênh AUROC TRAIN - TEST.'],
    luu_y='Chỉ ĐỌC mô hình đã huấn luyện, không chọn hay chỉnh gì theo TEST; AUROC TEST trùng Table A.',
    hoi=[('Có overfit không?', 'Chênh AUROC TRAIN - TEST: B0 +0.002, H1 khoảng 0, H2 +0.008 (nhỏ hơn độ dao động giữa seed ±0.016) => không overfit đáng kể.')])

A['so_sanh_early_stopping'] = dict(
    muc='Huấn luyện thêm bản H1 KHÔNG early stopping (đủ 150 epoch) cho mỗi seed, lưu 2 checkpoint riêng và so sánh với bản có early stopping.',
    cach=['Lưu trạng thái bộ sinh số ngẫu nhiên (random, numpy, torch).',
          'Lưu bản có early stopping (chính là H1 của Tables A–C) thành h1_seed<s>_early_stop.pt.',
          'Huấn luyện bản dung_som=False, lưu h1_seed<s>_no_early_stop.pt, dựng mạng Bayes H1 tương ứng.',
          'Tính AUROC, log-loss trên TRAIN/VAL/TEST và ATE trên TEST cho cả hai bản.',
          'Khôi phục trạng thái ngẫu nhiên để các cell sau (refuter, Table C) không đổi.'],
    hoi=[('Bỏ early stopping thì sao?', 'Loss VAL tăng (0.504 so với 0.500), AUROC TEST giảm nhẹ (0.799 so với 0.801): dấu hiệu bắt đầu overfit. ATE trung bình gần như không đổi nhưng từng seed lệch tới ±0.02.')])

A['ve_early_stopping'] = dict(
    muc='Hình 8.2b: loss TRAIN/VAL của bản H1 không dừng sớm qua 150 epoch, vạch dọc ở epoch bản có early stopping dừng, chấm ở trọng số được giữ.')

A['dowhy_refute_on_graph'] = dict(
    muc='Hai refuter DoWhy (Slide 69–71) trên ĐÚNG DAG của B0 hoặc H2, dùng ước lượng hỗ trợ hồi quy tuyến tính, cho Table C.',
    cach=['CausalModel với DAG của mô hình; identify_effect; estimate_effect backdoor.linear_regression.',
          'Placebo: hoán vị T (placebo_type="permute"), 30 lần -> hiệu ứng phải về khoảng 0.',
          'Subset: lấy ngẫu nhiên 80% dữ liệu, 30 lần -> hiệu ứng phải gần gốc.'],
    luu_y='Refuter này kiểm tra (đồ thị + ước lượng tuyến tính), không kiểm tra trực tiếp CPD BDeu hay MLP.')

A['h1_retraining_refuter'] = dict(
    muc='Kiểm định tương tự refuter cho H1 nơ-ron (vì refuter tuyến tính không áp dụng được): huấn luyện lại MLP trong 3 lần.',
    cach=['Placebo: hoán vị T trong TRAIN và VAL, huấn luyện lại MLP, tính ATE trên C_test -> kỳ vọng khoảng 0 (thực tế +0.018).',
          'Subset: huấn luyện lại trên 80% TRAIN, tính ATE trên C_test -> kỳ vọng gần gốc (+0.237 so với +0.245).',
          'Lấy trung bình 3 lần.'],
    luu_y='Được dán nhãn riêng trong Table C ("Custom retraining") để không nhầm với refuter DoWhy.')

A['doc_bang_csv'] = dict(muc='Đọc một bảng kết quả từ CSV trong artifacts/tables/.')

A['bang_hien_thi'] = dict(
    muc='Tạo BẢN SAO bảng để hiển thị, làm tròn 4 chữ số các ô số nằm trong cột kiểu chữ (cột trộn số với "mean ± std").',
    luu_y='Dữ liệu gốc không đổi vì các cell sau (forest plot) còn dùng.')

A['bang_positivity_T1_T2'] = dict(
    muc='Bảng 4 ô (học bổng x học phí đúng hạn) kèm tỷ lệ tốt nghiệp, để kiểm tra positivity của T3 và dấu hiệu nhân quả ngược của T2.',
    hoi=[('Bảng cho thấy gì?', 'Ô có học bổng nhưng chưa đóng đúng hạn chỉ 37 SV (positivity yếu); nhóm không học bổng và chưa đóng đúng hạn chỉ 4.5% tốt nghiệp (gợi ý bỏ học -> ngừng đóng phí).')])

A['doc_do_thi_da_luu'] = dict(muc='Đọc file JSON đồ thị do vòng lặp 5 run đã lưu (artifacts/graphs/<tên>.json).')

A['mat_na_thoi_gian'] = dict(
    muc='Dựng lại mặt nạ M để vẽ heatmap Hình 8.1(a): cấm tự nối, cấm Y -> mọi nút, cấm T -> Z.',
    luu_y='Cùng nội dung với build_domain_mask nhưng viết lại theo thứ tự nút của hình.')

A['bang_tom_tat_do_thi'] = dict(
    muc='Bảng tóm tắt đồ thị 5 seed: hàm điểm B0 được chọn, số cạnh, Pa(Y); tau, số cạnh, Pa(Y), h(W), hội tụ, có ép T -> Y không của H2.')

A['ma_hoa_7_bien'] = dict(
    muc='Mã hóa lại 7 biến đúng như mục 8.A (từ dữ liệu cột gốc) để mục 8.2c dùng lại split và checkpoint đã lưu.')

A['tai_mlp_h1'] = dict(
    muc='Dựng lại kiến trúc MLP của H1 và nạp trọng số từ checkpoint h1_seed42.pt (chế độ eval).',
    luu_y='Tạo Module().net để khớp tên khóa "net.*" trong state_dict đã lưu.')

A['ma_hoa_dau_vao_h1'] = dict(muc='Mã hóa Pa(Y) như H1 (Z1 one-hot 3 chiều, biến nhị phân giữ nguyên) cho phần tính tay ở 8.2c.')

A['q_b0'] = dict(
    muc='Tính tay q_B0(t, C_i): tra trực tiếp P(Y=1 | T=t, cha khác = giá trị của hồ sơ) trong bảng CPD của B0.',
    ly_thuyet='Minh bạch hóa công thức q_m(t, C_i) của Slide 59: đặt T = t cho mọi hồ sơ, giữ nguyên hiệp biến, tra bảng.')

A['q_h1'] = dict(muc='Tính tay q_H1(t, C_i): đặt T = t, mã hóa Pa(Y), cho MLP dự đoán và qua sigmoid.')

A['AdvancedConfig'] = dict(
    muc='Cấu hình riêng cho mục 9: đường dẫn dữ liệu, seed 42, 1000 bootstrap và 7 hiệp biến (thêm Displaced, Gender so với mục 8).',
    luu_y='Mục 9 dùng biến gốc (tuổi, điểm liên tục) và 7 hiệp biến, khác mã hóa rời rạc 5 biến của mục 8.')

A['__code_69'] = dict(
    muc='Mục 9.A: tạo hai quần thể: df_completed (3,630 SV có kết cục) và df_full (4,424 SV, gồm Enrolled) với kết cục giữ chân Y_retain = 1 nếu không bỏ học.',
    hoi=[('Vì sao cần quần thể 4,424?', 'Để kiểm tra kết luận không phụ thuộc việc loại Enrolled (mục 9.D).')])

A['__code_71'] = dict(
    muc='Mục 9.B: hiệu ứng của học bổng lên điểm trung bình HK2 bằng hồi quy OLS Y_grade ~ T + 7 hiệp biến.',
    cach=['Hệ số của T là ATE (thang 20): +1.48, CI [1.09, 1.86].', 'So với chênh lệch thô +2.63: điều chỉnh làm giảm gần một nửa.'],
    luu_y='Điểm HK2 là kết cục sau T, ở đây dùng làm KẾT CỤC (không phải biến điều chỉnh) nên hợp lệ.')

A['__code_73'] = dict(
    muc='Mục 9.C: hiệu ứng lên tỷ lệ đạt tín chỉ HK2 = số tín chỉ đạt / số đăng ký (mẫu số tối thiểu 1 để tránh chia 0), OLS với 7 hiệp biến.',
    cach=['ATE = +14.5 điểm %, CI [11.8%, 17.2%].'])

A['__code_75'] = dict(
    muc='Mục 9.D: hiệu ứng lên tỷ lệ giữ chân trên 4,424 SV bằng AIPW (doubly robust).',
    cach=['Mô hình propensity: logistic T ~ hiệp biến, chặn e(x) trong [0.05, 0.95] để tránh trọng số cực lớn.',
          'Hai mô hình kết cục: logistic riêng cho nhóm T=0 (mu0) và T=1 (mu1).',
          'Điểm AIPW: dr1 = mu1 + T(Y - mu1)/e; dr0 = mu0 + (1-T)(Y - mu0)/(1-e).',
          'ATE = trung bình(dr1) - trung bình(dr0) = +16.87%.',
          'CI: bootstrap 1000 lần trên các điểm dr (không khớp lại mô hình) -> [13.79%, 19.73%].'],
    ly_thuyet='Doubly robust: ước lượng nhất quán nếu MỘT TRONG HAI mô hình (propensity hoặc kết cục) đúng.',
    hoi=[('Vì sao +16.87% nhỏ hơn +24%?', 'Kết cục khác (giữ chân gồm cả đang học) và quần thể khác; điều quan trọng là vẫn dương rõ.')])

A['__code_77'] = dict(
    muc='Mục 9.E: tách hiệu ứng thành trực tiếp (NDE) và gián tiếp qua tín chỉ HK1 (NIE) bằng phương pháp tích hệ số.',
    cach=['Mô hình M ~ T + hiệp biến: hệ số a của T (+0.73 tín chỉ).', 'Mô hình Y ~ T + M + hiệp biến (LPM): NDE = hệ số của T, b = hệ số của M.',
          'NIE = a x b; tổng = NDE + NIE; tỷ lệ trung gian = NIE / tổng = 21.5%.'],
    ly_thuyet='Giả định thêm: không có nhiễu giữa M và Y, và mô hình tuyến tính không có tương tác T x M.',
    hoi=[('21.5% nghĩa là gì?', 'Khoảng một phần năm tác động của học bổng đi qua việc học tốt hơn ở HK1; phần lớn còn lại là trực tiếp (tài chính, động lực...).')])

A['__code_79'] = dict(
    muc='Mục 9.F: CATE theo 6 nhóm định trước (tuổi, Debtor, nơi ở) bằng G-computation logistic khớp riêng trong từng nhóm.',
    cach=['Trong mỗi nhóm: logistic Y ~ T + hiệp biến (bỏ hiệp biến hằng trong nhóm).', 'Đặt T=0 cho mọi người, dự đoán, lấy trung bình -> Risk(do(0)); tương tự với T=1.',
          'CATE = hiệu; cột "x" là tỷ số rủi ro Risk(1)/Risk(0).'],
    luu_y='Chưa có CI cho từng nhóm nên chênh lệch giữa các nhóm cần đọc thận trọng; "2.03x" là tỷ số rủi ro, không phải "hưởng lợi gấp đôi".')

A['KetQuaDoNhayCinelli'] = dict(muc='Dataclass kết quả độ nhạy: hệ số của T, t-stat, bậc tự do, partial R^2 của T với Y, RV và RV ở mức alpha 0.05.')

A['tinh_rv_cinelli_hazlett'] = dict(
    muc='Tính Robustness Value (Cinelli & Hazlett 2020) từ hồi quy OLS Y ~ T + Z1..Z5: một nhiễu ẩn phải mạnh cỡ nào mới xóa được hiệu ứng.',
    cach=['Mã hóa 5 Z (thứ tự riêng của mục 9.H: Z1 tuổi, Z2 điểm vào, Z3 học vấn mẹ, Z4 Debtor, Z5 học phí).',
          'Giải bình phương tối thiểu, tính sai số chuẩn và t-stat của T (t = 14.20, df = 3623).',
          'f = |t| / sqrt(df); RV = 0.5 x (sqrt(f^4 + 4 f^2) - f^2) = 20.97%.',
          'RV ở alpha 0.05: thay f bằng f - t*_{0.975, df-1}/sqrt(df-1) -> 18.37%.',
          'Partial R^2 của T với Y = t^2 / (t^2 + df) = 5.27%.'],
    ly_thuyet='RV = 21% nghĩa là nhiễu ẩn phải giải thích ít nhất khoảng 21% phương sai dư của CẢ T lẫn Y thì mới đưa ước lượng về 0.',
    hoi=[('21% là lớn hay nhỏ?', 'So với biến quan sát mạnh nhất là học phí (partial R^2 với Y 10.4%): kết luận vững trước nhiễu ẩn yếu hơn học phí, nhưng không miễn nhiễm với nhiễu mạnh gấp đôi.')])

A['benchmark_partial_r2'] = dict(
    muc='Partial R^2 của từng biến nền với Y (khi đã có T và các biến nền khác), làm mốc so sánh cho RV.',
    cach=['RSS đầy đủ: hồi quy Y ~ T + mọi Z.', 'Với mỗi Z: RSS khi bỏ Z đó.', 'Partial R^2 = (RSS bỏ Z - RSS đầy đủ) / RSS bỏ Z.'])

A['ma_hoa_z_chinh_sach'] = dict(muc='Mã hóa 5 biến nền cho mô phỏng phân bổ (cùng mã hóa với mục 9.H), kèm T và Y.')

A['danh_gia_chinh_sach'] = dict(
    muc='So sánh 4 cách phân bổ CÙNG số suất học bổng (969): theo CATE, thực tế, ưu tiên Debtor, ngẫu nhiên; ước lượng tỷ lệ tốt nghiệp dự kiến.',
    cach=['T-learner: logistic riêng cho T=0 (mu0) và T=1 (mu1) trên toàn bộ dữ liệu.',
          'CATE ước lượng = mu1 - mu0 cho từng SV.',
          'Mỗi chính sách là vector pi (1 = được cấp): CATE optimal lấy 969 người có CATE cao nhất; Need-based xếp theo Debtor rồi học phí.',
          'Giá trị chính sách V(pi) = trung bình[pi x mu1 + (1 - pi) x mu0]; so số SV tốt nghiệp với phân bổ thực tế.'],
    luu_y='In-sample: cùng mu0, mu1 vừa dùng để chọn vừa dùng để chấm, chưa cross-fitting nên +164 SV có xu hướng lạc quan. Xếp hạng còn dùng biến học phí (có thể là hậu quả của bỏ học).',
    hoi=[('+164 SV có đáng tin không?', 'Là ước lượng mô hình, không phải quan sát; cần cross-fitting và đánh giá off-policy (IPW/DR) mới chắc.')])

A['__code_52'] = dict(
    muc='Vòng lặp chính: với mỗi seed 42..46, khớp B0, H1, H2 trên cùng split rồi đánh giá bằng record_model_run.',
    cach=['B0: learn_b0_dag + fit_bn_on_dag, đo thời gian, đánh giá, lưu JSON đồ thị.',
          'H1: lấy Pa(Y) của B0, train_h1_mlp, build_h1_bayesian_network, lưu checkpoint, đánh giá.',
          'H2: train_notears_mlp với lambda đã chọn, select_h2_graph, fit_bn_on_dag, đánh giá, lưu W và JSON (kể cả log h(W) từng vòng).',
          'assert không có trọng số khác 0 ở ô bị cấm.',
          'Giữ mô hình, dữ liệu và nhật ký trong run_artifacts[seed] cho mục 8.E2.'],
    ly_thuyet='Ghép cặp: trong cùng một seed, ba mô hình dùng cùng TRAIN/VAL/TEST, nên hiệu B0 - H1 - H2 là so sánh theo cặp (Slide 68).')

A['__code_55'] = dict(
    muc='Xuất Table A (dự báo) và Table B (can thiệp): tính hiệu theo cặp so với B0 trong cùng seed, thêm dòng tổng hợp trung bình ± độ lệch chuẩn, ghi CSV.')

A['__code_57'] = dict(
    muc='Chạy 4 việc cho seed 42 rồi xuất Table C: (1) lấy sơ đồ B0, H2 và mạng B0 của seed 42; (2) refuter DoWhy trên B0 và H2; (3) refuter huấn luyện lại cho H1; (4) gom thành bảng Table C (độ bền vững + chi phí).',
    cach=['Lấy đúng TRAIN, VAL, TEST của seed 42 từ split đã đóng băng.',
          'Học lại sơ đồ B0 và lấy sơ đồ H2 đã chọn; điền bảng BDeu để có mạng Bayes B0.',
          'Hai refuter DoWhy (placebo và subset, mỗi loại 30 lần) trên sơ đồ B0 và H2.',
          'Refuter tự viết cho H1 (huấn luyện lại 3 lần).',
          'Gom kết quả ba mô hình thành một dòng mỗi mô hình, rồi ghi Table C ra CSV.'],
    ly_thuyet='Refuter chỉ kiểm tra độ ổn định của ước lượng; không chứng minh sơ đồ đúng hay không có nhiễu ẩn. Y_test không được dùng để tính ATE.',
)
