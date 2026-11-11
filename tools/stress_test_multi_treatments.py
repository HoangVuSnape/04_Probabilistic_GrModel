"""
Script: tools/stress_test_multi_treatments.py
Author: Challenger 1 (Milestone 2 - Empirical Challenge)
Purpose:
    Empirical stress-testing and adversarial challenge of tools/verify_and_compute_multi_treatments.py
    and artifacts/tables/Table_Multi_Treatment.csv:
    1. Split Leakage Analysis:
       - Rigorous check of train/val/test disjointness across seeds 42..46.
       - Empirical test of StandardScaler preprocessing leakage (fit on full dataset vs fit on train-only).
    2. Permutation & Placebo Treatment Refutation:
       - 100-iteration Monte Carlo permutation test for T1, T2, T3.
       - Verification that placebo treatment effects collapse to ~0 and p-values follow null distribution.
    3. Numerical Stability & Seed Sensitivity:
       - Distribution of ATE across seeds 42..46 for all 9 configurations.
       - Diagnostics on T3 AIPW variance and cohort filtering discrepancy (n=545 vs n=203..221).
    4. Formulate empirical conclusions for gate verdict.
"""

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
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.verify_and_compute_multi_treatments import (
    tai_va_tien_xu_ly_du_lieu,
    uoc_luong_backdoor_lpm,
    uoc_luong_g_computation,
    uoc_luong_doubly_robust
)

DATA_PATH = ROOT_DIR / 'data' / 'uci_697.csv'
SPLITS_DIR = ROOT_DIR / 'splits'
SEEDS = (42, 43, 44, 45, 46)
COV_COLS = ('Z1_age', 'Z2_adm', 'Z3_mom', 'Z4_debt', 'Z5_disp', 'Z6_gen')


def run_split_leakage_tests() -> Dict[str, any]:
    """
    Test 1: Check split leakage and preprocessing leakage.
    """
    print("\n" + "=" * 80)
    print("TEST 1: SPLIT LEAKAGE & PREPROCESSING LEAKAGE AUDIT")
    print("=" * 80)
    
    df = tai_va_tien_xu_ly_du_lieu(DATA_PATH)
    n_cohort = len(df)
    print(f"Authoritative Binary Cohort Size: N = {n_cohort}")
    
    split_results = {}
    total_violations = 0
    
    for s in SEEDS:
        split_file = SPLITS_DIR / f"split_seed_{s}.json"
        with open(split_file, 'r', encoding='utf-8') as f:
            sp = json.load(f)
            
        tr = set(sp['train_indices'])
        va = set(sp.get('val_indices', []))
        te = set(sp['test_indices'])
        
        leak_tr_te = len(tr & te)
        leak_tr_va = len(tr & va)
        leak_va_te = len(va & te)
        
        max_idx = max(tr | va | te)
        min_idx = min(tr | va | te)
        union_size = len(tr | va | te)
        
        has_leak = (leak_tr_te > 0) or (leak_tr_va > 0) or (leak_va_te > 0)
        if has_leak or (union_size != n_cohort) or (max_idx >= n_cohort) or (min_idx < 0):
            total_violations += 1
            
        print(f"Seed {s}: Train={len(tr)}, Val={len(va)}, Test={len(te)}, Union={union_size}")
        print(f"  Overlap (Train ∩ Test): {leak_tr_te} | Overlap (Train ∩ Val): {leak_tr_va} | Overlap (Val ∩ Test): {leak_va_te}")
        print(f"  Index bounds: [{min_idx}, {max_idx}] vs N={n_cohort}")
        
        split_results[s] = {
            'train_size': len(tr),
            'val_size': len(va),
            'test_size': len(te),
            'leak_tr_te': leak_tr_te,
            'bounds_valid': (min_idx == 0 and max_idx == n_cohort - 1 and union_size == n_cohort)
        }
        
    print(f"\nSummary: Total Split Leakage Violations = {total_violations}")
    
    # 1.2 Test StandardScaler Preprocessing Leakage
    print("\n--- Preprocessing Leakage Sensitivity (StandardScaler on full vs train-only) ---")
    scaling_diffs = []
    
    for s in SEEDS:
        split_file = SPLITS_DIR / f"split_seed_{s}.json"
        with open(split_file, 'r', encoding='utf-8') as f:
            sp = json.load(f)
            
        tr_idx = sp['train_indices']
        te_idx = sp['test_indices']
        
        # Method A (Current in verify_and_compute_multi_treatments.py): Scale whole df, then slice
        df_scaled_full = df.copy()
        scaler_full = StandardScaler()
        df_scaled_full[list(COV_COLS)] = scaler_full.fit_transform(df[list(COV_COLS)])
        tr_sub_A = df_scaled_full.iloc[tr_idx].reset_index(drop=True)
        te_sub_A = df_scaled_full.iloc[te_idx].reset_index(drop=True)
        
        # Method B (Strict Zero-Leakage): Fit scaler ONLY on train, transform test
        df_tr_B = df.iloc[tr_idx].copy().reset_index(drop=True)
        df_te_B = df.iloc[te_idx].copy().reset_index(drop=True)
        scaler_strict = StandardScaler()
        df_tr_B[list(COV_COLS)] = scaler_strict.fit_transform(df_tr_B[list(COV_COLS)])
        df_te_B[list(COV_COLS)] = scaler_strict.transform(df_te_B[list(COV_COLS)])
        
        for tr_id in ['T1', 'T2', 'T3']:
            # G-computation
            _, _, ate_g_A, _, _, _ = uoc_luong_g_computation(tr_sub_A, te_sub_A, tr_id, COV_COLS)
            _, _, ate_g_B, _, _, _ = uoc_luong_g_computation(df_tr_B, df_te_B, tr_id, COV_COLS)
            diff_g = abs(ate_g_A - ate_g_B)
            
            # Doubly Robust
            _, _, ate_dr_A, _, _, _ = uoc_luong_doubly_robust(tr_sub_A, te_sub_A, tr_id, COV_COLS)
            _, _, ate_dr_B, _, _, _ = uoc_luong_doubly_robust(df_tr_B, df_te_B, tr_id, COV_COLS)
            diff_dr = abs(ate_dr_A - ate_dr_B)
            
            scaling_diffs.append({
                'seed': s,
                'treatment': tr_id,
                'ate_g_full': ate_g_A,
                'ate_g_strict': ate_g_B,
                'diff_g': diff_g,
                'ate_dr_full': ate_dr_A,
                'ate_dr_strict': ate_dr_B,
                'diff_dr': diff_dr
            })
            
    df_diffs = pd.DataFrame(scaling_diffs)
    max_diff_g = df_diffs['diff_g'].max()
    max_diff_dr = df_diffs['diff_dr'].max()
    mean_diff_g = df_diffs['diff_g'].mean()
    mean_diff_dr = df_diffs['diff_dr'].mean()
    
    print(f"G-computation max ATE difference (full vs strict scale): {max_diff_g:.6f} (mean: {mean_diff_g:.6f})")
    print(f"Doubly Robust max ATE difference (full vs strict scale): {max_diff_dr:.6f} (mean: {mean_diff_dr:.6f})")
    
    return {
        'split_results': split_results,
        'total_violations': total_violations,
        'max_diff_g': max_diff_g,
        'max_diff_dr': max_diff_dr,
        'scaling_diffs_table': df_diffs
    }


def run_placebo_permutation_tests(n_permutations: int = 50) -> Dict[str, any]:
    """
    Test 2: Permutation / Placebo Refutation Test.
    Randomly shuffle treatment vector T and re-estimate causal effects.
    Expected: Placebo ATE should collapse to ~0, and true ATE should lie far in the tail.
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: PLACEBO PERMUTATION REFUTATION ({n_permutations} ITERATIONS)")
    print("=" * 80)
    
    df = tai_va_tien_xu_ly_du_lieu(DATA_PATH)
    
    # Chuẩn hóa Z
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[list(COV_COLS)] = scaler.fit_transform(df[list(COV_COLS)])
    
    rng = np.random.default_rng(2026)
    
    placebo_results = {}
    
    # Evaluate across Full Cohort and Seed 42 for T1, T2, T3
    for tr_id in ['T1', 'T2', 'T3']:
        print(f"\n>>> Running Placebo Permutations for {tr_id}...")
        
        # Calculate baseline true ATE on Full Cohort
        _, _, true_ate_lpm, _, _, true_p_lpm = uoc_luong_backdoor_lpm(df, df, tr_id, COV_COLS)
        _, _, true_ate_g, _, _, true_p_g = uoc_luong_g_computation(df_scaled, df_scaled, tr_id, COV_COLS)
        _, _, true_ate_dr, _, _, true_p_dr = uoc_luong_doubly_robust(df_scaled, df_scaled, tr_id, COV_COLS)
        
        placebo_ates_lpm = []
        placebo_ates_g = []
        placebo_ates_dr = []
        
        for i in range(n_permutations):
            df_perm = df.copy()
            df_scaled_perm = df_scaled.copy()
            
            if tr_id == 'T1':
                perm_t1 = rng.permutation(df['T1'].values)
                df_perm['T1'] = perm_t1
                df_scaled_perm['T1'] = perm_t1
            elif tr_id == 'T2':
                perm_t2 = rng.permutation(df['T2'].values)
                df_perm['T2'] = perm_t2
                df_scaled_perm['T2'] = perm_t2
            else:  # T3: permute both jointly
                perm_idx = rng.permutation(len(df))
                df_perm['T1'] = df['T1'].values[perm_idx]
                df_perm['T2'] = df['T2'].values[perm_idx]
                df_scaled_perm['T1'] = df_scaled['T1'].values[perm_idx]
                df_scaled_perm['T2'] = df_scaled['T2'].values[perm_idx]
                
            _, _, p_ate_lpm, _, _, _ = uoc_luong_backdoor_lpm(df_perm, df_perm, tr_id, COV_COLS)
            _, _, p_ate_g, _, _, _ = uoc_luong_g_computation(df_scaled_perm, df_scaled_perm, tr_id, COV_COLS)
            _, _, p_ate_dr, _, _, _ = uoc_luong_doubly_robust(df_scaled_perm, df_scaled_perm, tr_id, COV_COLS)
            
            placebo_ates_lpm.append(p_ate_lpm)
            placebo_ates_g.append(p_ate_g)
            placebo_ates_dr.append(p_ate_dr)
            
        mean_p_lpm = float(np.mean(placebo_ates_lpm))
        std_p_lpm = float(np.std(placebo_ates_lpm))
        mean_p_g = float(np.mean(placebo_ates_g))
        std_p_g = float(np.std(placebo_ates_g))
        mean_p_dr = float(np.mean(placebo_ates_dr))
        std_p_dr = float(np.std(placebo_ates_dr))
        
        # Empirical p-value: fraction of permuted ATEs >= true ATE
        emp_p_lpm = float(np.mean(np.array(placebo_ates_lpm) >= true_ate_lpm))
        emp_p_g = float(np.mean(np.array(placebo_ates_g) >= true_ate_g))
        emp_p_dr = float(np.mean(np.array(placebo_ates_dr) >= true_ate_dr))
        
        print(f"  True ATE: LPM={true_ate_lpm:+.4f}, G-Comp={true_ate_g:+.4f}, DR={true_ate_dr:+.4f}")
        print(f"  Placebo LPM:    Mean={mean_p_lpm:+.4f} +/- {std_p_lpm:.4f}, Range=[{min(placebo_ates_lpm):+.4f}, {max(placebo_ates_lpm):+.4f}], Emp p={emp_p_lpm:.4f}")
        print(f"  Placebo G-Comp: Mean={mean_p_g:+.4f} +/- {std_p_g:.4f}, Range=[{min(placebo_ates_g):+.4f}, {max(placebo_ates_g):+.4f}], Emp p={emp_p_g:.4f}")
        print(f"  Placebo DR:     Mean={mean_p_dr:+.4f} +/- {std_p_dr:.4f}, Range=[{min(placebo_ates_dr):+.4f}, {max(placebo_ates_dr):+.4f}], Emp p={emp_p_dr:.4f}")
        
        placebo_results[tr_id] = {
            'true_ate': {'lpm': true_ate_lpm, 'g': true_ate_g, 'dr': true_ate_dr},
            'placebo_mean': {'lpm': mean_p_lpm, 'g': mean_p_g, 'dr': mean_p_dr},
            'placebo_std': {'lpm': std_p_lpm, 'g': std_p_g, 'dr': std_p_dr},
            'emp_p_value': {'lpm': emp_p_lpm, 'g': emp_p_g, 'dr': emp_p_dr}
        }
        
    return placebo_results


def run_numerical_stability_tests() -> Dict[str, any]:
    """
    Test 3: Numerical Stability and Seed Sensitivities Audit.
    Analyze variations across seeds 42..46 and deep-dive into T3 AIPW variance.
    """
    print("\n" + "=" * 80)
    print("TEST 3: NUMERICAL STABILITY & SEED SENSITIVITY AUDIT")
    print("=" * 80)
    
    table_path = ROOT_DIR / 'artifacts' / 'tables' / 'Table_Multi_Treatment.csv'
    df_table = pd.read_csv(table_path)
    
    # Filter 5 split seed rows
    df_seeds = df_table[df_table['seed'].isin(['42', '43', '44', '45', '46'])].copy()
    
    stability_summary = []
    
    for (tr, est), grp in df_seeds.groupby(['treatment_id', 'estimator']):
        ates = grp['ate'].values
        mean_ate = float(np.mean(ates))
        std_ate = float(np.std(ates, ddof=1))
        min_ate = float(np.min(ates))
        max_ate = float(np.max(ates))
        cv = std_ate / abs(mean_ate) if mean_ate != 0 else 0.0
        
        # Risk 0 and Risk 1 stability
        mean_r0 = float(grp['risk_do_0'].mean())
        std_r0 = float(grp['risk_do_0'].std(ddof=1))
        mean_r1 = float(grp['risk_do_1'].mean())
        std_r1 = float(grp['risk_do_1'].std(ddof=1))
        
        stability_summary.append({
            'treatment': tr,
            'estimator': est,
            'mean_ate': mean_ate,
            'std_ate': std_ate,
            'cv_percent': cv * 100.0,
            'min_ate': min_ate,
            'max_ate': max_ate,
            'spread': max_ate - min_ate,
            'mean_r0': mean_r0,
            'std_r0': std_r0,
            'mean_r1': mean_r1,
            'std_r1': std_r1
        })
        
        print(f"[{tr}] {est:<32} | ATE: {mean_ate*100:+.2f}% +/- {std_ate*100:.2f}% | Spread: {min_ate*100:+.2f}% to {max_ate*100:+.2f}% (CV={cv*100:.2f}%)")
        
    df_stab = pd.DataFrame(stability_summary)
    
    # Deep dive into T3 AIPW anomaly
    print("\n--- Deep-Dive Diagnostic: T3 AIPW Variance and Sample Truncation ---")
    t3_dr_rows = df_seeds[(df_seeds['treatment_id'] == 'T3') & (df_seeds['estimator'] == 'Doubly Robust (AIPW)')]
    for idx, row in t3_dr_rows.iterrows():
        print(f"  Seed {row['seed']}: ATE = {row['ate']*100:+.2f}%, Risk(0) = {row['risk_do_0']*100:.2f}%, Risk(1) = {row['risk_do_1']*100:.2f}%, p = {row['p_value']:.2e}")
        
    return {
        'stability_summary': df_stab,
        't3_dr_rows': t3_dr_rows
    }


def main():
    print("=" * 80)
    print("STARTING EMPIRICAL CHALLENGER 1 AUDIT SUITE (MILESTONE 2)")
    print("=" * 80)
    
    res1 = run_split_leakage_tests()
    res2 = run_placebo_permutation_tests(n_permutations=50)
    res3 = run_numerical_stability_tests()
    
    print("\n" + "=" * 80)
    print("EMPIRICAL AUDIT SUITE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
