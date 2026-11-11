# -*- coding: utf-8 -*-
"""Thêm chú thích # cho đoạn code Table C (cell 57) vào annotated_code.py và fn_annotations.py."""
from pathlib import Path

CODE = r'''ref_seed = 42  # seed tham chiếu cho Table C (seed đầu tiên của 5 run)
split = CACHED_SPLITS[ref_seed]  # chỉ số TRAIN/VAL/TEST đã đóng băng của seed 42
tr42 = PRIMARY_DF.iloc[split['train_indices']].reset_index(drop=True)  # 2,541 SV để học
va42 = PRIMARY_DF.iloc[split['val_indices']].reset_index(drop=True)  # 544 SV để chọn thiết lập
te42 = PRIMARY_DF.iloc[split['test_indices']].reset_index(drop=True)  # 545 SV TEST (chỉ dùng để chấm cuối; không dùng Y để tính ATE)
b0_dag42, _, _ = learn_b0_dag(tr42, va42, GLOBAL_CONFIG)  # học lại sơ đồ B0 của seed 42 (hai giá trị đi kèm bỏ qua bằng _)
b0_bn42 = fit_bn_on_dag(tr42, b0_dag42, GLOBAL_CONFIG)  # điền bảng BDeu trên TRAIN để có mạng Bayes B0 đầy đủ
h2_dag42, _, _ = select_h2_graph(run_artifacts[ref_seed]['H2_res'], tr42, va42, GLOBAL_CONFIG)  # sơ đồ H2 đã chọn ngưỡng tau của seed 42 (lấy từ kết quả mục 8.E)

print('Đang chạy refuter DoWhy trên DAG B0 và DAG H2 (seed 42, TRAIN)...')  # thông báo tiến trình
ref_b0 = dowhy_refute_on_graph(b0_dag42, tr42, GLOBAL_CONFIG, ref_seed)  # placebo + subset trên sơ đồ B0 (dict gồm original, placebo, subset, backdoor)
ref_h2 = dowhy_refute_on_graph(h2_dag42, tr42, GLOBAL_CONFIG, ref_seed)  # cùng hai refuter trên sơ đồ H2
print('Đang chạy kiểm định huấn luyện lại cho H1 (placebo + subset)...')  # thông báo tiến trình
ref_h1 = h1_retraining_refuter(b0_bn42, tr42, va42, te42, GLOBAL_CONFIG, ref_seed)  # H1 không dùng DoWhy được nên huấn luyện lại MLP với T xáo / 80% TRAIN
h1_orig = float(df_interv_raw[(df_interv_raw['Model'] == 'H1') & (df_interv_raw['Seed'] == ref_seed)]['ATE'].iloc[0])  # ATE gốc của H1 seed 42 (lấy từ Table B để so với refuter)

refuter_map = {  # gom kết quả của ba mô hình để lặp ở bước sau
    'B0': {'Refuter_Target': f"DoWhy LR on B0 DAG, backdoor {{{', '.join(ref_b0['backdoor'])}}}",  # tên refuter + tập điều chỉnh của B0
           'Original': ref_b0['original'], 'Placebo': ref_b0['placebo'], 'Subset': ref_b0['subset'],  # ATE gốc, ATE sau placebo, ATE sau subset
           'Sims': GLOBAL_CONFIG.placebo_simulations},  # số lần mô phỏng (30)
    'H1': {'Refuter_Target': 'Custom retraining of H1 MLP (permuted T / 80% TRAIN), ATE on C_test',  # H1 dùng refuter tự viết, không phải DoWhy
           'Original': h1_orig, 'Placebo': ref_h1['placebo'], 'Subset': ref_h1['subset'],
           'Sims': GLOBAL_CONFIG.h1_refute_sims},  # số lần huấn luyện lại (3)
    'H2': {'Refuter_Target': f"DoWhy LR on H2 DAG, backdoor {{{', '.join(ref_h2['backdoor'])}}}",  # tương tự B0 nhưng trên sơ đồ H2
           'Original': ref_h2['original'], 'Placebo': ref_h2['placebo'], 'Subset': ref_h2['subset'],
           'Sims': GLOBAL_CONFIG.placebo_simulations},
}

table_c_rows = []  # mỗi mô hình một dòng của Table C
for m in ['B0', 'H1', 'H2']:  # lặp qua ba mô hình
    sub_p = df_pred_raw[df_pred_raw['Model'] == m]  # các dòng dự báo (Table A) của mô hình m, 5 seed
    sub_i = df_interv_raw[df_interv_raw['Model'] == m]  # các dòng can thiệp (Table B) của mô hình m, 5 seed
    r = refuter_map[m]  # kết quả refuter của mô hình m
    table_c_rows.append({
        'Model': m, 'Refuter_Target': r['Refuter_Target'], 'Refuter_Seed': ref_seed, 'Simulations': r['Sims'],
        'Original_Effect': f"{r['Original']:+.4f}",  # ATE gốc, có dấu +/-
        'Placebo_Effect': f"{r['Placebo']:+.4f} (Pass: {abs(r['Placebo']) < 0.05})",  # đạt nếu |ATE placebo| < 0.05 (gần 0)
        'Subset_Effect': f"{r['Subset']:+.4f} (Pass: {abs(r['Subset'] - r['Original']) < 0.05})",  # đạt nếu ATE subset lệch gốc < 0.05
        'ATE_Mean_±_Std_5_Runs': mean_std(sub_i['ATE'], '+.4f'),  # ATE trung bình ± độ lệch chuẩn qua 5 seed
        'Paired_Spread_Max_Min': f"{sub_i['ATE'].max() - sub_i['ATE'].min():.4f}",  # độ tản: max trừ min qua 5 seed
        'Delta_ATE_vs_B0': mean_std(sub_i['Delta_ATE_vs_B0'], '+.4f') if m != 'B0' else '---',  # chênh so với B0 cùng seed (B0 để ---)
        'Fit_Time_mean_s': f"{sub_p['Fit_Time_s'].mean():.2f}", 'Query_Time_mean_s': f"{sub_p['Query_Time_s'].mean():.4f}",  # thời gian khớp và truy vấn trung bình
        'Hardware': HARDWARE,  # cấu hình máy để tái lập
    })
df_table_c = pd.DataFrame(table_c_rows)  # chuyển danh sách dòng thành bảng
df_table_c.to_csv(Path(GLOBAL_CONFIG.tables_dir) / 'Table_C_Robustness_Cost.csv', index=False)  # ghi file CSV Table C
print('[Xuất bản] Table C đã lưu.')  # xác nhận đã ghi file
'''

NOTE = '''
A['__code_57'] = dict(
    muc='Chạy 4 việc cho seed 42 rồi xuất Table C: (1) lấy sơ đồ B0, H2 và mạng B0 của seed 42; (2) refuter DoWhy trên B0 và H2; (3) refuter huấn luyện lại cho H1; (4) gom thành bảng Table C (độ bền vững + chi phí).',
    cach=['Lấy đúng TRAIN, VAL, TEST của seed 42 từ split đã đóng băng.',
          'Học lại sơ đồ B0 và lấy sơ đồ H2 đã chọn; điền bảng BDeu để có mạng Bayes B0.',
          'Hai refuter DoWhy (placebo và subset, mỗi loại 30 lần) trên sơ đồ B0 và H2.',
          'Refuter tự viết cho H1 (huấn luyện lại 3 lần).',
          'Gom kết quả ba mô hình thành một dòng mỗi mô hình, rồi ghi Table C ra CSV.'],
    ly_thuyet='Refuter chỉ kiểm tra độ ổn định của ước lượng; không chứng minh sơ đồ đúng hay không có nhiễu ẩn. Y_test không được dùng để tính ATE.',
)
'''

p = Path('tools/so_tay/annotated_code.py')
p.write_text(p.read_text(encoding='utf-8') + "\nA['__code_57'] = r'''" + CODE + "'''\n", encoding='utf-8')

p = Path('tools/so_tay/fn_annotations.py')
s = p.read_text(encoding='utf-8')
old = "('fn', 'h1_retraining_refuter'), ('fn', 'doc_bang_csv')"
assert s.count(old) == 1, 'không tìm thấy mốc trong GROUPS'
s = s.replace(old, "('fn', 'h1_retraining_refuter'), ('code', 57, 'chay_refuter_va_table_c'), ('fn', 'doc_bang_csv')")
s += NOTE
p.write_text(s, encoding='utf-8')
print('đã thêm chú thích cell 57')
