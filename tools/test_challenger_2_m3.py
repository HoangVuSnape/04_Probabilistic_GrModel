"""
Challenger 2 Milestone 3 Gate Verification Script
Empirical adversarial testing and cross-checking for:
- 00_MASTER_Causal_PGM_Educational_Intervention.ipynb
- notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb
- report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md
"""

import json
import re
import sys
import hashlib
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def check_task1():
    print("=" * 80)
    print("TASK 1: Cross-check Naive Associational Gap across 4 sources & check eradication")
    print("=" * 80)
    
    nb_path = Path("00_MASTER_Causal_PGM_Educational_Intervention.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    qa_path = Path("report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md")
    qa_text = qa_path.read_text(encoding="utf-8")
    
    cells = nb["cells"]
    print(f"Total cells in master notebook: {len(cells)}")
    
    # 1. Cell 05 code output
    c5 = cells[5]
    c5_output_text = ""
    for out in c5.get("outputs", []):
        if "text" in out:
            c5_output_text += "".join(out["text"])
        elif "data" in out and "text/plain" in out["data"]:
            c5_output_text += "".join(out["data"]["text/plain"])
    
    c5_has_p_y1_t1 = "86.17%" in c5_output_text
    c5_has_p_y1_t0 = "51.63%" in c5_output_text
    c5_has_gap = "+34.54%" in c5_output_text
    c5_has_n = "969" in c5_output_text and "2,661" in c5_output_text
    print(f"Cell 05 output check: P(Y=1|T=1)=86.17%: {c5_has_p_y1_t1}, P(Y=1|T=0)=51.63%: {c5_has_p_y1_t0}, Gap=+34.54%: {c5_has_gap}, N counts: {c5_has_n}")
    
    # 2. Cell 06 markdown text
    c6_source = "".join(cells[6].get("source", []))
    c6_has_p_y1_t1 = "86.17" in c6_source and "835" in c6_source and "969" in c6_source
    c6_has_p_y1_t0 = "51.63" in c6_source and "1,374" in c6_source and "2,661" in c6_source
    c6_has_gap = "+34.54" in c6_source
    print(f"Cell 06 text check: P(Y=1|T=1)=86.17% (835/969): {c6_has_p_y1_t1}, P(Y=1|T=0)=51.63% (1374/2661): {c6_has_p_y1_t0}, Gap=+34.54%: {c6_has_gap}")
    
    # 3. Cell 30 Table 11.1
    c30_source = "".join(cells[30].get("source", []))
    c30_has_table11 = "11.1" in c30_source
    c30_has_gap = "+34.54" in c30_source
    print(f"Cell 30 Table 11.1 check: Has Table 11.1: {c30_has_table11}, Gap=+34.54%: {c30_has_gap}")
    
    # 4. QA Script Vòng 2 and Summary Table
    vong2_match = re.search(r'##\s+VÒNG\s+2:.*?(?=##\s+VÒNG\s+3:)', qa_text, re.DOTALL | re.IGNORECASE)
    vong2_text = vong2_match.group(0) if vong2_match else ""
    v2_has_p_y1_t1 = "86.17" in vong2_text and "835" in vong2_text and "969" in vong2_text
    v2_has_p_y1_t0 = "51.63" in vong2_text and "1,374" in vong2_text and "2,661" in vong2_text
    v2_has_gap = "+34.54" in vong2_text
    print(f"QA Script Vòng 2 check: P(Y=1|T=1)=86.17% (835/969): {v2_has_p_y1_t1}, P(Y=1|T=0)=51.63% (1374/2661): {v2_has_p_y1_t0}, Gap=+34.54%: {v2_has_gap}")
    
    summary_match = re.search(r'##\s+BẢNG\s+TỔNG\s+HỢP\s+CÁC\s+CON\s+SỐ.*', qa_text, re.DOTALL | re.IGNORECASE)
    summary_text = summary_match.group(0) if summary_match else ""
    sum_has_gap = "+34.54" in summary_text and "86.17" in summary_text and "51.63" in summary_text
    sum_has_counts = "835/969" in summary_text and "1,374/2,661" in summary_text and "3,630" in summary_text
    print(f"QA Script Summary Table check: Gap & Proportions: {sum_has_gap}, Raw Counts (835/969 vs 1,374/2,661, N=3,630): {sum_has_counts}")
    
    # 5. Eradication of stale values
    stale_patterns = [
        r'33\.55%?',
        r'33\.90%?',
        r'33\.9%',
        r'38\.09%?',
        r'76\.06%?',
        r'37\.97%?',
        r'85\.45%?',
        r'51\.90%?',
        r'828\s*/\s*969',
        r'1[,.]?381\s*/\s*2[,.]?661',
    ]
    
    full_nb_text = json.dumps(nb, ensure_ascii=False)
    nb_stale_found = {}
    for pat in stale_patterns:
        matches = re.findall(pat, full_nb_text)
        if matches:
            nb_stale_found[pat] = len(matches)
            print(f"WARNING: Found {len(matches)} matches for pattern '{pat}' in notebook!")
    if not nb_stale_found:
        print("PASSED: Zero stale values in notebook.")
        
    qa_stale_found = {}
    for pat in stale_patterns:
        matches = re.findall(pat, qa_text)
        if matches:
            qa_stale_found[pat] = len(matches)
            print(f"WARNING: Found {len(matches)} matches for pattern '{pat}' in QA script!")
    if not qa_stale_found:
        print("PASSED: Zero stale values in QA script.")
        
    task1_passed = all([
        c5_has_p_y1_t1, c5_has_p_y1_t0, c5_has_gap, c5_has_n,
        c6_has_p_y1_t1, c6_has_p_y1_t0, c6_has_gap,
        c30_has_table11, c30_has_gap,
        v2_has_p_y1_t1, v2_has_p_y1_t0, v2_has_gap,
        sum_has_gap, sum_has_counts,
        len(nb_stale_found) == 0, len(qa_stale_found) == 0
    ])
    print(f"\nTASK 1 OVERALL STATUS: {'PASSED' if task1_passed else 'FAILED'}")
    return task1_passed

def check_task2():
    print("\n" + "=" * 80)
    print("TASK 2: Cross-check Cell 11 parameter learning code and markdown (ESS = 10.0)")
    print("=" * 80)
    
    nb_path = Path("00_MASTER_Causal_PGM_Educational_Intervention.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    
    c11 = cells[11]
    c11_src = "".join(c11.get("source", []))
    has_ess_10_code = "equivalent_sample_size: float = 10.0" in c11_src and "equivalent_sample_size=equivalent_sample_size" in c11_src
    has_no_ess_5 = "equivalent_sample_size=5" not in c11_src and "equivalent_sample_size = 5" not in c11_src
    print(f"Cell 11 code check: has ESS=10.0: {has_ess_10_code}, has NO ESS=5: {has_no_ess_5}")
    
    c12_src = "".join(cells[12].get("source", []))
    md_ess_10 = "ESS = 10" in c12_src or "10.0" in c12_src
    print(f"Accompanying markdown (Cell 12) mentions ESS 10: {md_ess_10}")
    
    qa_path = Path("report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md")
    qa_text = qa_path.read_text(encoding="utf-8")
    vong4_match = re.search(r'##\s+VÒNG\s+4:.*?(?=##\s+VÒNG\s+5:)', qa_text, re.DOTALL | re.IGNORECASE)
    vong4_text = vong4_match.group(0) if vong4_match else ""
    v4_has_ess_10 = "10.0" in vong4_text or "ESS = 10" in vong4_text
    v4_has_no_ess_5 = "ESS = 5" not in vong4_text and "ESS=5" not in vong4_text
    print(f"QA Script Vòng 4 check: has ESS=10.0: {v4_has_ess_10}, has NO ESS=5: {v4_has_no_ess_5}")
    
    task2_passed = has_ess_10_code and has_no_ess_5 and md_ess_10 and v4_has_ess_10 and v4_has_no_ess_5
    print(f"\nTASK 2 OVERALL STATUS: {'PASSED' if task2_passed else 'FAILED'}")
    return task2_passed

def check_task3():
    print("\n" + "=" * 80)
    print("TASK 3: Cross-check Cell 22 loaded tables (Table_Factorial_H3.csv & Table_Multi_Treatment.csv)")
    print("=" * 80)
    
    nb_path = Path("00_MASTER_Causal_PGM_Educational_Intervention.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    c22 = nb["cells"][22]
    c22_src = "".join(c22.get("source", []))
    
    has_factorial_h3 = "Table_Factorial_H3.csv" in c22_src and "display(df_table_h3)" in c22_src
    has_multi_treatment = "Table_Multi_Treatment.csv" in c22_src and "display(df_table_multi_t)" in c22_src
    
    f_h3 = Path("artifacts/tables/Table_Factorial_H3.csv")
    f_multi = Path("artifacts/tables/Table_Multi_Treatment.csv")
    print(f"Cell 22 source loads and displays Table_Factorial_H3.csv: {has_factorial_h3}")
    print(f"Cell 22 source loads and displays Table_Multi_Treatment.csv: {has_multi_treatment}")
    print(f"artifacts/tables/Table_Factorial_H3.csv exists: {f_h3.exists()}")
    print(f"artifacts/tables/Table_Multi_Treatment.csv exists: {f_multi.exists()}")
    
    task3_passed = has_factorial_h3 and has_multi_treatment and f_h3.exists() and f_multi.exists()
    print(f"\nTASK 3 OVERALL STATUS: {'PASSED' if task3_passed else 'FAILED'}")
    return task3_passed

def check_task4():
    print("\n" + "=" * 80)
    print("TASK 4: Verify QA Script Vòng 7 Multi-Treatment metrics")
    print("=" * 80)
    
    qa_path = Path("report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md")
    qa_text = qa_path.read_text(encoding="utf-8")
    
    vong7_match = re.search(r'##\s+VÒNG\s+7:.*?(?=##\s+BẢNG\s+TỔNG\s+HỢP|###\s+BẢNG\s+TỔNG\s+HỢP|$)', qa_text, re.DOTALL | re.IGNORECASE)
    if not vong7_match:
        print("FAILED: Could not find VÒNG 7 in QA script!")
        return False
        
    v7_text = vong7_match.group(0)
    
    # 1. T1 (Scholarship) ATE ~ +21.8%
    has_t1 = "21.8" in v7_text
    
    # 2. T2 (Tuition debt relief) ATE ~ +50.1% (p < 1e-50)
    has_t2 = "50.1" in v7_text
    has_p_1e50 = "10^{-55}" in v7_text or "1e-50" in v7_text or "10^{-50}" in v7_text
    
    # 3. T3 (Joint policy) ATE ~ +73.8% (Risk(1)=84.55%)
    has_t3 = "73.8" in v7_text
    has_risk1 = "84.55" in v7_text
    
    # Verify Risk(0)=9.52% and Risk(1)=84.55% in Table_Multi_Treatment.csv
    df_multi = pd.read_csv("artifacts/tables/Table_Multi_Treatment.csv")
    t3_pooled = df_multi[(df_multi['treatment_id'] == 'T3') & (df_multi['estimator'] == 'Doubly Robust (AIPW)') & (df_multi['seed'] == 'Pooled 5 Runs')]
    has_t3_risk0 = False
    has_t3_risk1 = False
    if not t3_pooled.empty:
        r0 = float(t3_pooled['risk_do_0'].values[0])
        r1 = float(t3_pooled['risk_do_1'].values[0])
        has_t3_risk0 = abs(r0 - 0.095221) < 0.001
        has_t3_risk1 = abs(r1 - 0.845522) < 0.001
        print(f"Table_Multi_Treatment.csv Pooled AIPW T3: Risk(0)={r0:.4f} (9.52%), Risk(1)={r1:.4f} (84.55%)")
        
    print(f"T1 (+21.8%): {has_t1}")
    print(f"T2 (+50.1%): {has_t2}, p < 1e-50: {has_p_1e50}")
    print(f"T3 (+73.8%): {has_t3}, Risk(1)=84.55%: {has_risk1}")
    print(f"Table underlying Risk(0)=9.52%: {has_t3_risk0}, Risk(1)=84.55%: {has_t3_risk1}")
    
    task4_passed = has_t1 and has_t2 and has_p_1e50 and has_t3 and has_risk1 and has_t3_risk0 and has_t3_risk1
    print(f"\nTASK 4 OVERALL STATUS: {'PASSED' if task4_passed else 'FAILED'}")
    return task4_passed

def check_parity():
    print("\n" + "=" * 80)
    print("EXTRA: Mirror Parity and Integrity")
    print("=" * 80)
    
    p1 = Path("00_MASTER_Causal_PGM_Educational_Intervention.ipynb")
    p2 = Path("notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb")
    
    h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    identical = (h1 == h2)
    print(f"Root notebook SHA256:   {h1}")
    print(f"Mirror notebook SHA256: {h2}")
    print(f"Bit-for-bit identical:  {identical}")
    return identical

if __name__ == "__main__":
    t1 = check_task1()
    t2 = check_task2()
    t3 = check_task3()
    t4 = check_task4()
    tp = check_parity()
    
    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY: T1={t1}, T2={t2}, T3={t3}, T4={t4}, Parity={tp}")
    all_pass = all([t1, t2, t3, t4, tp])
    print(f"OVERALL VERDICT: {'CONFIRM' if all_pass else 'REJECT'}")
    print("=" * 80)
    sys.exit(0 if all_pass else 1)
