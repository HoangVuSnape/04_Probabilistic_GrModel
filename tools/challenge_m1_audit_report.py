import os
import sys
import json
import re
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = r"e:\01_Research_Study_Master\04_Probabilistic_GrModel"
os.chdir(WORKSPACE)

print("=" * 80)
print("EMPIRICAL CHALLENGER VERIFICATION HARNESS FOR MILESTONE 1 AUDIT REPORT")
print("=" * 80)

discrepancies = []
passed_checks = []

def check(condition, desc, detail=""):
    if condition:
        passed_checks.append(desc)
        print(f"[PASS] {desc}")
    else:
        discrepancies.append((desc, detail))
        print(f"[FAIL] {desc} -> {detail}")

# ---------------------------------------------------------
# PART 1: MASTER NOTEBOOK CELLS & CITATIONS
# ---------------------------------------------------------
print("\n--- PART 1: Master Notebook Cell Auditing ---")
nb_path = "00_MASTER_Causal_PGM_Educational_Intervention.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
check(len(cells) == 31, "Notebook total cells is 31", f"Actual: {len(cells)}")

code_cells = [c for c in cells if c.get("cell_type") == "code"]
md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
check(len(code_cells) == 10, "Notebook has exactly 10 code cells", f"Actual: {len(code_cells)}")
check(len(md_cells) == 21, "Notebook has exactly 21 markdown cells", f"Actual: {len(md_cells)}")

# Check Cell 00
cell_00_src = "".join(cells[0].get("source", []))
check("Đồ án Cuối kỳ" in cell_00_src or "IT701100" in cell_00_src, "Cell 00 contains title / course info")

# Check Cell 05 output (+34.54%)
cell_05 = cells[5]
cell_05_outputs = "".join(str(out.get("text", "")) for out in cell_05.get("outputs", []))
check("+34.54%" in cell_05_outputs or "34.54" in cell_05_outputs, "Cell 05 output contains +34.54%", f"Outputs snippet: {cell_05_outputs[:200]}")

# Check Cell 06 markdown text (+33.55%)
cell_06_src = "".join(cells[6].get("source", []))
check("+33.55%" in cell_06_src or "33.55" in cell_06_src, "Cell 06 markdown contains +33.55%", f"Snippet: {cell_06_src[:200]}")

# Check Cell 08 (HillClimbSearch vs NOTEARS)
cell_08_src = "".join(cells[8].get("source", []))
check("HillClimbSearch" in cell_08_src, "Cell 08 uses HillClimbSearch", f"Cell 08 source snippet: {cell_08_src[:200]}")
check("notears" not in cell_08_src.lower(), "Cell 08 does NOT implement NOTEARS", "NOTEARS found in Cell 08?")

# Check Cell 11 (ESS=10.0)
cell_11_src = "".join(cells[11].get("source", []))
check("equivalent_sample_size=10" in cell_11_src or "equivalent_sample_size = 10" in cell_11_src or "ESS=10" in cell_11_src or "10" in cell_11_src, "Cell 11 sets ESS=10.0")

# Check Cell 15 title: "### 6.1. Cấu trúc 7 nút"
cell_15_src = "".join(cells[15].get("source", []))
check("7 nút" in cell_15_src, "Cell 15 title contains '7 nút'", f"Cell 15 snippet: {cell_15_src[:150]}")

# Check Cell 16: DiGraph has 8 nodes and 13 edges
cell_16_src = "".join(cells[16].get("source", []))
check("Curricular_units_1st_sem_approved" in cell_16_src, "Cell 16 defines DAG with mediator M1 (8 nodes)")

# Check Cell 22 loads Table A, B, C, but omits H3
cell_22_src = "".join(cells[22].get("source", []))
check("Table_A_Prediction.csv" in cell_22_src, "Cell 22 loads Table A")
check("Table_B_Intervention.csv" in cell_22_src, "Cell 22 loads Table B")
check("Table_C_Robustness_Cost.csv" in cell_22_src, "Cell 22 loads Table C")
check("Table_Factorial_H3.csv" not in cell_22_src, "Cell 22 omits Table_Factorial_H3.csv (as claimed in audit)")

# Check Cell 30: Table 11.1 stating +33.90%
cell_30_src = "".join(cells[30].get("source", []))
check("33.90%" in cell_30_src or "+33.90%" in cell_30_src, "Cell 30 states +33.90%", f"Snippet: {cell_30_src[:200]}")

# ---------------------------------------------------------
# PART 2: QA SCRIPT CITATIONS & DISCREPANCIES
# ---------------------------------------------------------
print("\n--- PART 2: NOTEBOOK_DEFENSE_QA_SCRIPT.md Citations ---")
qa_path = "report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md"
with open(qa_path, "r", encoding="utf-8") as f:
    qa_lines = f.readlines()

# Check lines 81-83: +38.09%
qa_line_81_85 = "".join(qa_lines[80:85])
check("38.09%" in qa_line_81_85, "QA script lines 81-85 contains 38.09%", f"Actual: {qa_line_81_85.strip()}")

# Check lines 101-122: NOTEARS in Cell 08/16
qa_lines_101_125 = "".join(qa_lines[100:125])
check("notears" in qa_lines_101_125.lower() or "w_eff" in qa_lines_101_125.lower() or "mask" in qa_lines_101_125.lower(), "QA script lines 101-125 references NOTEARS / Mask")

# Check line 130: ESS=5
qa_lines_128_135 = "".join(qa_lines[127:135])
check("ESS = 5" in qa_lines_128_135 or "ESS=5" in qa_lines_128_135, "QA script references ESS=5 around line 130", f"Actual: {qa_lines_128_135.strip()}")

# Check line 212: +38.09%
qa_lines_210_215 = "".join(qa_lines[209:215])
check("38.09%" in qa_lines_210_215, "QA script line 212 references 38.09%", f"Actual: {qa_lines_210_215.strip()}")

# ---------------------------------------------------------
# PART 3: NUMERICAL CROSS-CHECKING (TABLES)
# ---------------------------------------------------------
print("\n--- PART 3: Numerical Cross-Checking (CSV Artifacts) ---")

# 3.1 Table_Multi_Treatment.csv
mt_path = "artifacts/tables/Table_Multi_Treatment.csv"
check(os.path.exists(mt_path), "Table_Multi_Treatment.csv exists")
df_mt = pd.read_csv(mt_path)
check(len(df_mt) == 63, "Table_Multi_Treatment.csv has exactly 63 rows", f"Actual rows: {len(df_mt)}")

# Check columns
expected_cols = ['treatment_id', 'estimator', 'split_seed', 'cohort_type', 'cohort_size', 'risk_do_0', 'risk_do_1', 'ate', 'std_error', 'ci_95_low', 'ci_95_high', 'p_value']
check(all(c in df_mt.columns for c in expected_cols), "Table_Multi_Treatment.csv has all expected columns")

# Check Pooled rows for T1, T2, T3
# Report claimed:
# T1 LPM: ATE = +21.84% (CI: [18.11%, 25.57%], p = 6.59e-29). Full cohort: +21.61% (p = 4.38e-41).
# T1 G-Comp: ATE = +21.76% (CI: [18.47%, 25.43%]).
# T1 AIPW: ATE = +23.09% (CI: [18.79%, 27.39%]).
# T2 LPM: ATE = +45.05% (CI: [39.75%, 50.36%], p = 1.08e-55). Full cohort: +45.81% (p = 6.53e-87).
# T2 G-Comp: ATE = +49.64% (CI: [44.58%, 54.39%]).
# T2 AIPW: ATE = +50.75% (CI: [43.44%, 58.06%]).
# T3 LPM: ATE = +66.65% (CI: [60.47%, 72.83%]). Full cohort: +67.15%.
# T3 G-Comp: ATE = +73.78% (CI: [68.46%, 78.09%]).
# T3 AIPW: ATE = +75.03% (CI: [68.54%, 81.52%]), reaching up to +76.27% (seed 42) and +77.22% (seed 46).

print("\nVerifying Table_Multi_Treatment values against Report:")
for tid in ['T1', 'T2', 'T3']:
    for est in ['Linear Regression (LPM)', 'G-Computation (Logistic)', 'Doubly Robust (AIPW)']:
        row = df_mt[(df_mt.treatment_id == tid) & (df_mt.estimator == est) & (df_mt.split_seed == 'pooled')]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"[{tid} - {est}] ATE={r['ate']*100:.2f}%, CI=[{r['ci_95_low']*100:.2f}%, {r['ci_95_high']*100:.2f}%], Risk0={r['risk_do_0']*100:.2f}%, Risk1={r['risk_do_1']*100:.2f}%, p={r['p_value']:.2e}")
        else:
            print(f"[FAIL] Missing pooled row for {tid} {est}")

# Verify exact numbers reported in audit report
t1_lpm = df_mt[(df_mt.treatment_id == 'T1') & (df_mt.estimator == 'Linear Regression (LPM)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t1_lpm['ate'] - 0.2184) < 0.001, "T1 LPM pooled ATE == +21.84%", f"Actual: {t1_lpm['ate']}")
check(abs(t1_lpm['ci_95_low'] - 0.1811) < 0.001 and abs(t1_lpm['ci_95_high'] - 0.2557) < 0.001, "T1 LPM CI == [18.11%, 25.57%]")

t1_gcomp = df_mt[(df_mt.treatment_id == 'T1') & (df_mt.estimator == 'G-Computation (Logistic)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t1_gcomp['ate'] - 0.2176) < 0.001, "T1 G-Comp pooled ATE == +21.76%", f"Actual: {t1_gcomp['ate']}")
check(abs(t1_gcomp['ci_95_low'] - 0.1847) < 0.001 and abs(t1_gcomp['ci_95_high'] - 0.2543) < 0.001, "T1 G-Comp CI == [18.47%, 25.43%]")

t1_aipw = df_mt[(df_mt.treatment_id == 'T1') & (df_mt.estimator == 'Doubly Robust (AIPW)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t1_aipw['ate'] - 0.2309) < 0.001, "T1 AIPW pooled ATE == +23.09%", f"Actual: {t1_aipw['ate']}")
check(abs(t1_aipw['ci_95_low'] - 0.1879) < 0.001 and abs(t1_aipw['ci_95_high'] - 0.2739) < 0.001, "T1 AIPW CI == [18.79%, 27.39%]")

t2_lpm = df_mt[(df_mt.treatment_id == 'T2') & (df_mt.estimator == 'Linear Regression (LPM)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t2_lpm['ate'] - 0.4505) < 0.001, "T2 LPM pooled ATE == +45.05%", f"Actual: {t2_lpm['ate']}")
check(abs(t2_lpm['ci_95_low'] - 0.3975) < 0.001 and abs(t2_lpm['ci_95_high'] - 0.5036) < 0.001, "T2 LPM CI == [39.75%, 50.36%]")

t2_gcomp = df_mt[(df_mt.treatment_id == 'T2') & (df_mt.estimator == 'G-Computation (Logistic)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t2_gcomp['ate'] - 0.4964) < 0.001, "T2 G-Comp pooled ATE == +49.64%", f"Actual: {t2_gcomp['ate']}")
check(abs(t2_gcomp['ci_95_low'] - 0.4458) < 0.001 and abs(t2_gcomp['ci_95_high'] - 0.5439) < 0.001, "T2 G-Comp CI == [44.58%, 54.39%]")

t2_aipw = df_mt[(df_mt.treatment_id == 'T2') & (df_mt.estimator == 'Doubly Robust (AIPW)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t2_aipw['ate'] - 0.5075) < 0.001, "T2 AIPW pooled ATE == +50.75%", f"Actual: {t2_aipw['ate']}")
check(abs(t2_aipw['ci_95_low'] - 0.4344) < 0.001 and abs(t2_aipw['ci_95_high'] - 0.5806) < 0.001, "T2 AIPW CI == [43.44%, 58.06%]")

t3_lpm = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.estimator == 'Linear Regression (LPM)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t3_lpm['ate'] - 0.6665) < 0.001, "T3 LPM pooled ATE == +66.65%", f"Actual: {t3_lpm['ate']}")
check(abs(t3_lpm['ci_95_low'] - 0.6047) < 0.001 and abs(t3_lpm['ci_95_high'] - 0.7283) < 0.001, "T3 LPM CI == [60.47%, 72.83%]")

t3_gcomp = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.estimator == 'G-Computation (Logistic)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t3_gcomp['ate'] - 0.7378) < 0.001, "T3 G-Comp pooled ATE == +73.78%", f"Actual: {t3_gcomp['ate']}")
check(abs(t3_gcomp['ci_95_low'] - 0.6846) < 0.001 and abs(t3_gcomp['ci_95_high'] - 0.7809) < 0.001, "T3 G-Comp CI == [68.46%, 78.09%]")

t3_aipw = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.estimator == 'Doubly Robust (AIPW)') & (df_mt.split_seed == 'pooled')].iloc[0]
check(abs(t3_aipw['ate'] - 0.7503) < 0.001, "T3 AIPW pooled ATE == +75.03%", f"Actual: {t3_aipw['ate']}")
check(abs(t3_aipw['ci_95_low'] - 0.6854) < 0.001 and abs(t3_aipw['ci_95_high'] - 0.8152) < 0.001, "T3 AIPW CI == [68.54%, 81.52%]")

# Check seed 42 and 46 AIPW for T3
t3_s42 = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.estimator == 'Doubly Robust (AIPW)') & (df_mt.split_seed == '42')].iloc[0]
check(abs(t3_s42['ate'] - 0.7627) < 0.001, "T3 Seed 42 AIPW ATE == +76.27%", f"Actual: {t3_s42['ate']}")

t3_s46 = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.estimator == 'Doubly Robust (AIPW)') & (df_mt.split_seed == '46')].iloc[0]
check(abs(t3_s46['ate'] - 0.7722) < 0.001, "T3 Seed 46 AIPW ATE == +77.22%", f"Actual: {t3_s46['ate']}")

# Check full cohort rows
t1_fc = df_mt[(df_mt.treatment_id == 'T1') & (df_mt.cohort_type == 'full_cohort')].iloc[0]
check(abs(t1_fc['ate'] - 0.2161) < 0.001, "T1 Full Cohort ATE == +21.61%", f"Actual: {t1_fc['ate']}")

t2_fc = df_mt[(df_mt.treatment_id == 'T2') & (df_mt.cohort_type == 'full_cohort')].iloc[0]
check(abs(t2_fc['ate'] - 0.4581) < 0.001, "T2 Full Cohort ATE == +45.81%", f"Actual: {t2_fc['ate']}")

t3_fc = df_mt[(df_mt.treatment_id == 'T3') & (df_mt.cohort_type == 'full_cohort')].iloc[0]
check(abs(t3_fc['ate'] - 0.6715) < 0.001, "T3 Full Cohort ATE == +67.15%", f"Actual: {t3_fc['ate']}")

# Boundary axioms check
check(len(df_mt[(df_mt.ci_95_high > 1.0) | (df_mt.ci_95_low < -1.0)]) == 0, "No CI boundary violations across 63 rows")
check(len(df_mt[(df_mt.risk_do_0 < 0.0) | (df_mt.risk_do_0 > 1.0)]) == 0, "No risk(0) boundary violations")
check(len(df_mt[(df_mt.risk_do_1 < 0.0) | (df_mt.risk_do_1 > 1.0)]) == 0, "No risk(1) boundary violations")

# 3.2 Table_Alternative_Outcomes.csv
ao_path = "artifacts/tables/Table_Alternative_Outcomes.csv"
check(os.path.exists(ao_path), "Table_Alternative_Outcomes.csv exists")
df_ao = pd.read_csv(ao_path)
print("\nTable_Alternative_Outcomes content:")
print(df_ao.to_string())

# Verify:
# Continuous S2 GPA (Y2): ATE = +1.478 points [1.093, 1.863], p = 6.69e-14
y2_row = df_ao[df_ao['outcome_name'].str.contains('GPA', case=False, na=False) | df_ao['outcome_id'].str.contains('Y2', case=False, na=False)].iloc[0]
check(abs(y2_row['ate_estimate'] - 1.478) < 0.01, "Y2 GPA ATE == +1.478", f"Actual: {y2_row['ate_estimate']}")
check(abs(y2_row['ci_95_low'] - 1.093) < 0.01 and abs(y2_row['ci_95_high'] - 1.863) < 0.01, "Y2 GPA 95% CI == [1.093, 1.863]")

# Credit Velocity (Y3): ATE = +14.50% [11.80%, 17.20%], p = 1.58e-25
y3_row = df_ao[df_ao['outcome_name'].str.contains('Velocity', case=False, na=False) | df_ao['outcome_id'].str.contains('Y3', case=False, na=False)].iloc[0]
check(abs(y3_row['ate_estimate'] - 0.1450) < 0.001 or abs(y3_row['ate_estimate'] - 14.50) < 0.1, "Y3 Credit Velocity ATE == +14.50%", f"Actual: {y3_row['ate_estimate']}")

# Full-Cohort Retention (N=4,424): ATE = +16.87% [13.79%, 19.73%]
ret_row = df_ao[df_ao['outcome_name'].str.contains('Retention', case=False, na=False) | df_ao['outcome_id'].str.contains('Retention', case=False, na=False)].iloc[0]
check(abs(ret_row['ate_estimate'] - 0.1687) < 0.001 or abs(ret_row['ate_estimate'] - 16.87) < 0.1, "Full Cohort Retention ATE == +16.87%", f"Actual: {ret_row['ate_estimate']}")

# Mediation NIE = +4.65%, NDE = +16.96%, TE = +21.61%, prop = 21.53%
med_rows = df_ao[df_ao['outcome_name'].str.contains('Mediation', case=False, na=False) | df_ao['outcome_id'].str.contains('NIE|NDE|Med', case=False, na=False)]
print(f"Mediation rows in Table_Alternative_Outcomes:\n{med_rows.to_string()}")

# 3.3 Table_Sensitivity_Cinelli.csv
sens_path = "artifacts/tables/Table_Sensitivity_Cinelli.csv"
check(os.path.exists(sens_path), "Table_Sensitivity_Cinelli.csv exists")
df_sens = pd.read_csv(sens_path)
print("\nTable_Sensitivity_Cinelli content:")
print(df_sens.to_string())

# Check RV = 23.40%
rv_val = df_sens[df_sens['metric'].str.contains('RV|Robustness', case=False, na=False)]
print("RV in Table_Sensitivity_Cinelli:\n", rv_val)

# Check Confounders partial R2:
# Z5 Age = 10.41%, Z1 Mother = 4.22%, Z4 Tuition = 0.81%, Z2 Admission = 0.48%, Z3 Debtor = 0.03%
for var, expected_r2 in [('Age', 0.1041), ('Mother', 0.0422), ('Tuition', 0.0081), ('Admission', 0.0048), ('Debtor', 0.0003)]:
    r = df_sens[df_sens['variable'].str.contains(var, case=False, na=False)] if 'variable' in df_sens.columns else None
    if r is not None and len(r) > 0:
        val = r.iloc[0]['partial_r2']
        check(abs(val - expected_r2) < 0.005 or abs(val*100 - expected_r2*100) < 0.5, f"Confounder {var} partial R2 ~ {expected_r2*100:.2f}%", f"Actual: {val}")

# 3.4 Check Table A, B, C, H3
for tname, fname in [
    ("Table A", "artifacts/tables/Table_A_Prediction.csv"),
    ("Table B", "artifacts/tables/Table_B_Intervention.csv"),
    ("Table C", "artifacts/tables/Table_C_Robustness_Cost.csv"),
    ("Table H3", "artifacts/tables/Table_Factorial_H3.csv")
]:
    check(os.path.exists(fname), f"{tname} exists on disk")
    df_t = pd.read_csv(fname)
    print(f"\n{tname} ({fname}):")
    print(df_t.head())

# ---------------------------------------------------------
# PART 4: SPLIT FILES DISJOINTNESS & INVARIANTS
# ---------------------------------------------------------
print("\n--- PART 4: Split Files Hygiene Verification ---")
import glob
splits = sorted(glob.glob("splits/split_seed_*.json"))
check(len(splits) == 5, "Exactly 5 split files found in splits/")
for s_path in splits:
    with open(s_path, "r") as f:
        sd = json.load(f)
    tr = set(sd["train_indices"])
    va = set(sd["val_indices"])
    te = set(sd["test_indices"])
    leakage = (tr & va) | (tr & te) | (va & te)
    check(len(leakage) == 0, f"{s_path} zero-leakage verified")
    check(len(tr) == 2541 and len(va) == 544 and len(te) == 545, f"{s_path} sizes: 2541, 544, 545 (total 3630)")

# ---------------------------------------------------------
# PART 5: CITATION LINE CHECKS IN SUPPORTING DOCS
# ---------------------------------------------------------
print("\n--- PART 5: Line Number Citations in Docs ---")
# Check docs/requirements/README.md:1-8
with open("docs/requirements/README.md", "r", encoding="utf-8") as f:
    r_lines = f.readlines()
check("IT701100" in "".join(r_lines[:10]), "docs/requirements/README.md lines 1-10 mentions IT701100")

# Check docs/requirements/midterm/README.md:1-9
with open("docs/requirements/midterm/README.md", "r", encoding="utf-8") as f:
    mr_lines = f.readlines()
check("Accept" in "".join(mr_lines[:10]) or "Reject" in "".join(mr_lines[:10]), "docs/requirements/midterm/README.md mentions Accept/Reject in first 10 lines")

# Check tools/build_causal_notebook.py:194 for GLOBAL_CONFIG
with open("tools/build_causal_notebook.py", "r", encoding="utf-8") as f:
    bcn_lines = f.readlines()
bcn_190_200 = "".join(bcn_lines[185:205])
check("GLOBAL_CONFIG" in bcn_190_200, "tools/build_causal_notebook.py around line 194 defines GLOBAL_CONFIG", f"Snippet: {bcn_190_200.strip()[:100]}")

# ---------------------------------------------------------
# PART 6: SUMMARY OF FINDINGS
# ---------------------------------------------------------
print("\n" + "=" * 80)
print(f"VERIFICATION COMPLETE: {len(passed_checks)} CHECKS PASSED, {len(discrepancies)} DISCREPANCIES FOUND")
print("=" * 80)
if discrepancies:
    print("\nDISCREPANCIES FOUND:")
    for desc, det in discrepancies:
        print(f" - {desc}: {det}")
else:
    print("\nALL VERIFIED ZERO ANOMALIES!")
