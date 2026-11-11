# -*- coding: utf-8 -*-
"""
NHÓM 05 — Cấu hình chung, mã hóa 7 biến và 5 split dùng chung (mục 8.A)

Mọi siêu tham số của B0/H1/H2 nằm trong Config; dữ liệu được mã hóa thành Z1..Z5, T, Y; 5 split 70/15/15 phân
tầng được lưu ra JSON và đóng băng.

Code được TRÍCH NGUYÊN VĂN từ notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm
chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).
Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.
File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook
(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).

Nội dung:
  - Config                                     Cell 41 (class)
  - init_directories                           Cell 41 (def)
  - set_seed                                   Cell 41 (def)
  - load_and_preprocess_uci697                 Cell 42 (def)
  - get_or_create_splits                       Cell 43 (def)
"""


# ==============================================================================================================
# Config  (class)
# Vị trí: Cell 41 · mục 8.A Cấu hình chung, dữ liệu 7 biến và 5 split
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Dataclass "đóng băng" chứa MỌI cấu hình của mục 8: seed, tỷ lệ split, tên nút, thư mục, siêu tham
#            số B0/H1/H2, tham số đánh giá.
# CÁCH LÀM :
#   1. seeds 42..46 cho 5 run ghép cặp; split 70/15/15.
#   2. B0: ESS=5, lưới hàm điểm {BIC, BDeu, K2}, cạnh bắt buộc T -> Y.
#   3. H1: 32 nơ-ron ẩn, dropout 0.2, lr 0.005, weight decay 1e-4, batch 64, tối đa 150 epoch, patience 15.
#   4. H2: 10 nơ-ron ẩn/nút, lưới lambda_L1 {1e-3, 1e-4}, lưới tau {0.1, 0.2, 0.3, 0.5, 0.8}, lr 0.01, 300 bước
#      trong, tối đa 12 vòng ngoài, h_tol 1e-6.
#   5. Đánh giá: 1000 bootstrap, 30 lần mô phỏng refuter, subset 80%.
# LƯU Ý    : frozen=True: không sửa được sau khi tạo, tránh vô tình đổi cấu hình giữa chừng.
# ❓ HỎI   : Siêu tham số chọn thế nào?
#   ➜ ĐÁP : Hàm điểm B0, tau và lambda của H2, ngưỡng F1 đều chọn trên VALIDATION; phần còn lại cố định trước,
#           không dò theo TEST.
# ==============================================================================================================
@dataclass(frozen=True)
class Config:
    # Hạt giống ngẫu nhiên cho 5 paired runs
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    
    # Tỷ lệ phân chia tập dữ liệu
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Nguồn dữ liệu UCI 697
    data_url: str = 'https://archive.ics.uci.edu/static/public/697/data.csv'
    data_local_path: str = 'data/uci_697.csv'
    
    # Danh mục biến số trong Primary DAG (7 nút)
    confounders: Tuple[str, ...] = ('Z1', 'Z2', 'Z3', 'Z4', 'Z5')
    treatment: str = 'T'
    outcome: str = 'Y'
    all_nodes: Tuple[str, ...] = ('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'T', 'Y')
    
    # Thư mục lưu trữ artifacts
    artifacts_dir: str = 'artifacts'
    splits_dir: str = 'artifacts/splits'
    tables_dir: str = 'artifacts/tables'
    checkpoints_dir: str = 'artifacts/checkpoints'
    graphs_dir: str = 'artifacts/graphs'
    
    # Siêu tham số Mô hình B0 (học DAG truyền thống + CPD BDeu)
    b0_ess: int = 5  # Equivalent Sample Size cho BDeu prior (dùng chung cho B0 và H2)
    b0_score_grid: Tuple[str, ...] = ('BIC', 'BDeu', 'K2')  # chọn trên VALIDATION
    required_edges: Tuple[Tuple[str, str], ...] = (('T', 'Y'),)  # giả thuyết cần ước lượng, áp cho cả B0 và H2

    # Siêu tham số Mô hình H1 (Neural CPD)
    h1_hidden_dim: int = 32
    h1_dropout: float = 0.2
    h1_lr: float = 0.005
    h1_weight_decay: float = 1e-4
    h1_batch_size: int = 64
    h1_epochs: int = 150
    h1_patience: int = 15
    h1_refute_sims: int = 3

    # Siêu tham số Mô hình H2 (NOTEARS-MLP phi tuyến)
    h2_hidden: int = 10
    h2_lambda_grid: Tuple[float, ...] = (1e-3, 1e-4)  # chọn trên VALIDATION của seed 42, rồi đóng băng
    h2_threshold_grid: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8)  # chọn trên VALIDATION từng seed
    h2_lr: float = 0.01
    h2_inner_iters: int = 300
    h2_max_outer: int = 12
    h2_h_tol: float = 1e-6
    h2_rho_max: float = 1e+16

    # Tham số đánh giá & kiểm định
    n_bootstrap: int = 1000
    placebo_simulations: int = 30
    subset_fraction: float = 0.80


# ==============================================================================================================
# init_directories  (def)
# Vị trí: Cell 41 · mục 8.A Cấu hình chung, dữ liệu 7 biến và 5 split
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tạo các thư mục artifacts/ (splits, tables, checkpoints, graphs) nếu chưa có.
# ==============================================================================================================
def init_directories(cfg: Config):
    for p in [cfg.artifacts_dir, cfg.splits_dir, cfg.tables_dir, cfg.checkpoints_dir, cfg.graphs_dir, 'data', 'splits']:
        Path(p).mkdir(parents=True, exist_ok=True)


# ==============================================================================================================
# set_seed  (def)
# Vị trí: Cell 41 · mục 8.A Cấu hình chung, dữ liệu 7 biến và 5 split
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Cố định mọi nguồn ngẫu nhiên: random, numpy, torch (và CUDA nếu có), PYTHONHASHSEED.
# LƯU Ý    : Gọi ở đầu mỗi lần huấn luyện để cùng seed thì ra cùng kết quả.
# ==============================================================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================================================================================================
# load_and_preprocess_uci697  (def)
# Vị trí: Cell 42 · mục 8.A Cấu hình chung, dữ liệu 7 biến và 5 split
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Nạp UCI 697 và mã hóa đúng 7 biến của mô hình chính: Z1..Z5, T, Y (mã hóa dùng cho Tables A–C).
# CÁCH LÀM :
#   1. Đọc data/uci_697.csv (nếu thiếu thì tải từ UCI và lưu cache).
#   2. Chuẩn hóa tên cột bằng cách bỏ ký tự đặc biệt rồi tra bảng alias (xử lý BOM, dấu nháy).
#   3. Loại Enrolled: 4,424 -> 3,630.
#   4. Z1 = điểm vào 3 mức (<120 -> 0, [120,140) -> 1, >=140 -> 2).
#   5. Z2 = Debtor, Z3 = học phí đúng hạn (0/1).
#   6. Z4 = học vấn mẹ >= 13 (đại học trở lên) -> 1.
#   7. Z5 = tuổi nhập học > 20 -> 1.
#   8. T = Scholarship holder; Y = Graduate 1 / Dropout 0.
#   9. assert không có NaN.
# LƯU Ý    : Bẫy ký hiệu: ở mục 8, Z1 = điểm vào, Z2 = Debtor, Z3 = học phí, Z4 = học vấn mẹ, Z5 = tuổi. Mục 6–7
#            và 9.H đánh số khác.
# ==============================================================================================================
def load_and_preprocess_uci697(cfg: Config) -> pd.DataFrame:
    """
    Nạp dữ liệu từ UCI hoặc cache nội bộ, lọc cohort Dropout/Graduate,
    binarize Outcome và tạo 7 biến số theo đúng hợp đồng nhân quả.
    """
    local_p = Path(cfg.data_local_path)
    if local_p.exists():
        print(f'Nạp dữ liệu từ cache cục bộ: {local_p}')
        df_raw = pd.read_csv(local_p, sep=None, engine='python')
    else:
        print(f'Tải dữ liệu từ UCI Repository: {cfg.data_url}')
        try:
            df_raw = pd.read_csv(cfg.data_url, sep=None, engine='python')
        except Exception:
            df_raw = pd.read_csv(cfg.data_url, sep=';')
        # Lưu cache
        local_p.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(local_p, index=False, sep=';')
    
    # Chuẩn hóa tên cột để xử lý BOM và ký tự đặc biệt
    aliases = {
        'admissiongrade': 'Admission grade',
        'ageatenrollment': 'Age at enrollment',
        'scholarshipholder': 'Scholarship holder',
        'target': 'Target',
        'debtor': 'Debtor',
        'tuitionfeesuptodate': 'Tuition fees up to date',
        'mothersqualification': "Mother's qualification",
    }
    renamed = {}
    for c in df_raw.columns:
        k = ''.join(ch for ch in str(c).lower() if ch.isalnum())
        renamed[c] = aliases.get(k, str(c).strip())
    df_raw = df_raw.rename(columns=renamed)
    
    # 1. Lọc bỏ nhóm 'Enrolled' (loại bỏ trạng thái trung gian chưa rõ kết quả)
    initial_count = len(df_raw)
    df_clean = df_raw[df_raw['Target'].isin(['Dropout', 'Graduate'])].copy()
    filtered_count = len(df_clean)
    print(f'[Eligibility Audit] Ban đầu: {initial_count} dòng -> Sau khi loại Enrolled: {filtered_count} dòng (Loại bỏ {initial_count - filtered_count} dòng).')
    
    # 2. Xây dựng 7 biến số Primary DAG
    df_model = pd.DataFrame(index=df_clean.index)
    
    # Z1: Admission grade (3 nhóm: < 120 -> 0; 120-140 -> 1; >= 140 -> 2)
    adm = pd.to_numeric(df_clean['Admission grade'], errors='coerce')
    df_model['Z1'] = pd.cut(adm, bins=[-np.inf, 120.0, 140.0, np.inf], labels=[0, 1, 2]).astype(int)
    
    # Z2: Debtor (0/1)
    df_model['Z2'] = df_clean['Debtor'].astype(int)
    
    # Z3: Tuition fees up to date (0/1)
    df_model['Z3'] = df_clean['Tuition fees up to date'].astype(int)
    
    # Z4: Mother's qualification (>= 13 là Đại học/Sau đại học -> 1, còn lại -> 0)
    mq = pd.to_numeric(df_clean["Mother's qualification"], errors='coerce')
    df_model['Z4'] = (mq >= 13).astype(int)
    
    # Z5: Age at enrollment (> 20 -> 1, <= 20 -> 0)
    age = pd.to_numeric(df_clean['Age at enrollment'], errors='coerce')
    df_model['Z5'] = (age > 20).astype(int)
    
    # T: Treatment - Scholarship holder (0/1)
    df_model['T'] = df_clean['Scholarship holder'].astype(int)
    
    # Y: Outcome - Target (Graduate = 1, Dropout = 0)
    df_model['Y'] = df_clean['Target'].map({'Graduate': 1, 'Dropout': 0}).astype(int)
    
    # Sắp xếp lại thứ tự cột chuẩn xác
    cols = list(cfg.all_nodes)
    df_model = df_model[cols].reset_index(drop=True)
    
    # Kiểm tra missing values
    assert df_model.isnull().sum().sum() == 0, 'Lỗi: Tồn tại giá trị NaN sau tiền xử lý!'
    print(f'[Sanity Check] Kích thước tập dữ liệu hoàn chỉnh: {df_model.shape}')
    print('[Class Balance] Phân bố nhãn Outcome Y:', df_model['Y'].value_counts().to_dict())
    print('[Treatment Balance] Phân bố can thiệp T:', df_model['T'].value_counts().to_dict())
    return df_model


# ==============================================================================================================
# get_or_create_splits  (def)
# Vị trí: Cell 43 · mục 8.A Cấu hình chung, dữ liệu 7 biến và 5 split
# --------------------------------------------------------------------------------------------------------------
# MỤC ĐÍCH : Tạo (hoặc nạp lại) 5 phân chia TRAIN/VAL/TEST 70/15/15 cho 5 seed, phân tầng theo tổ hợp (Y, T),
#            lưu JSON.
# CÁCH LÀM :
#   1. Tầng phân chia = chuỗi "Y_T" để tỷ lệ tốt nghiệp và học bổng giống nhau ở mọi tập.
#   2. Nếu file split đã có thì đọc lại (đóng băng).
#   3. Nếu chưa: train_test_split tách 70% / 30%, rồi tách 30% thành 15% VAL / 15% TEST (đều stratify).
#   4. Lưu chỉ số ra artifacts/splits/split_seed_<s>.json.
#   5. assert 3 tập không giao nhau và phủ đủ 3,630 dòng.
# LÝ THUYẾT: Slide 56: dùng chung TRAIN/VAL/TEST cho mọi mô hình; C_test (545 hồ sơ) là quần thể tham chiếu để
#            chuẩn hóa can thiệp.
# ❓ HỎI   : Vì sao 5 seed?
#   ➜ ĐÁP : Slide 39 yêu cầu ít nhất 5 run ghép cặp; B0/H1/H2 dùng cùng split trong mỗi seed nên so sánh được
#           theo cặp.
# ==============================================================================================================
def get_or_create_splits(df: pd.DataFrame, cfg: Config) -> Dict[int, Dict[str, List[int]]]:
    """
    Tạo hoặc nạp lại phân chia dữ liệu 70/15/15 cho 5 seeds.
    Phân tầng (Stratified) dựa trên tổ hợp của Y và T để đảm bảo cân bằng can thiệp.
    """
    splits_dict = {}
    n_samples = len(df)
    strata = df['Y'].astype(str) + '_' + df['T'].astype(str)
    
    for seed in cfg.seeds:
        split_file = Path(cfg.splits_dir) / f'split_seed_{seed}.json'
        alt_split_file = Path('splits') / f'split_seed_{seed}.json'
        
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
        elif alt_split_file.exists():
            with open(alt_split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
        else:
            set_seed(seed)
            indices = np.arange(n_samples)
            
            # Bước 1: Tách 70% Train và 30% Temp (Val + Test)
            train_idx, temp_idx = train_test_split(
                indices, test_size=(cfg.val_ratio + cfg.test_ratio),
                random_state=seed, stratify=strata
            )
            
            # Bước 2: Tách 30% Temp thành 15% Val và 15% Test
            temp_strata = strata.iloc[temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5,
                random_state=seed, stratify=temp_strata
            )
            
            split_data = {
                'seed': seed,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'test_indices': test_idx.tolist()
            }
            
            # Lưu file json
            for target_path in [split_file, alt_split_file]:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2)
        
        # Kiểm tra tính toàn vẹn (Sanity checks)
        tr_set = set(split_data['train_indices'])
        va_set = set(split_data['val_indices'])
        te_set = set(split_data['test_indices'])
        assert len(tr_set.intersection(va_set)) == 0, f'Seed {seed}: Train và Val bị giao nhau!'
        assert len(tr_set.intersection(te_set)) == 0, f'Seed {seed}: Train và Test bị giao nhau!'
        assert len(va_set.intersection(te_set)) == 0, f'Seed {seed}: Val và Test bị giao nhau!'
        assert len(tr_set) + len(va_set) + len(te_set) == n_samples, f'Seed {seed}: Không phủ hết tập mẫu!'
        
        splits_dict[seed] = split_data
        print(f'Seed {seed}: Train = {len(tr_set)} | Val = {len(va_set)} | Test = {len(te_set)}')
    
    return splits_dict
