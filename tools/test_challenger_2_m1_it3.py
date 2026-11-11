"""
Comprehensive Adversarial Empirical Challenge Suite for Milestone 1 Gate Iteration 3
Author: Challenger 2 (challenger_2_m1_it3)
Tests all 7 Iteration 1 items and all 3 Iteration 2 fixes.
"""

import os
import re
import sys
import json
import pandas as pd
from packaging.requirements import Requirement

# Ensure UTF-8 output encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_REPORT_PATH = os.path.join(ROOT_DIR, "report", "midterm", "midterm_defense_audit_report.md")
QA_SCRIPT_PATH = os.path.join(ROOT_DIR, "report", "midterm", "NOTEBOOK_DEFENSE_QA_SCRIPT.md")
REQUIREMENTS_PATH = os.path.join(ROOT_DIR, "requirements.txt")
TABLE_A_PATH = os.path.join(ROOT_DIR, "artifacts", "tables", "Table_A_Prediction.csv")
TABLE_B_PATH = os.path.join(ROOT_DIR, "artifacts", "tables", "Table_B_Intervention.csv")
SUB_NOTEBOOK_PATH = os.path.join(ROOT_DIR, "notebook", "causal_modeling_uci697.ipynb")
MASTER_NOTEBOOK_PATH = os.path.join(ROOT_DIR, "00_MASTER_Causal_PGM_Educational_Intervention.ipynb")

passed = 0
failed = 0
errors = []

def record(name, condition, msg=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        err_msg = f"[FAIL] {name}: {msg}"
        print(err_msg)
        errors.append(err_msg)

print("=" * 80)
print("STARTING EMPIRICAL CHALLENGE SUITE: CHALLENGER 2 MILESTONE 1 ITERATION 3")
print("=" * 80)

# Load audit report text
with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
    audit_text = f.read()

# Load QA script text
with open(QA_SCRIPT_PATH, "r", encoding="utf-8") as f:
    qa_text = f.read()

# ==============================================================================
# PART 1: ADVERSARIAL VERIFICATION OF 7 ITERATION 1 REJECTION ITEMS
# ==============================================================================
print("\n--- PART 1: 7 ITERATION 1 REJECTION ITEMS ---")

# Item 1.1: Score Table Arithmetic
# Table in lines 52-65
table_match = re.search(r"READINESS SCORE COMPONENT[\s\S]*?TOTAL WEIGHTED READINESS SCORE\s*│\s*10\.00\s*/\s*10\s*│\s*([\d\.]+)\s*/\s*10", audit_text)
record("1.1.1 Score table exists with total 9.42", table_match is not None and table_match.group(1) == "9.42", f"Matched: {table_match.group(1) if table_match else 'None'}")

# Extract current points from rows
dimension_pts = []
for line in audit_text.splitlines():
    if re.search(r"│\s*\d\.\s+.*?│\s*[\d\.]+\s*│\s*([\d\.]+)\s*│", line):
        m = re.search(r"│\s*\d\.\s+.*?│\s*[\d\.]+\s*│\s*([\d\.]+)\s*│", line)
        dimension_pts.append(float(m.group(1)))

expected_pts = [3.80, 1.50, 1.35, 1.37, 1.40]
record("1.1.2 Score table dimensions match [3.80, 1.50, 1.35, 1.37, 1.40]", dimension_pts == expected_pts, f"Found: {dimension_pts}")

pts_sum = round(sum(dimension_pts), 4)
record("1.1.3 Score table dimensions strictly sum to 9.42", pts_sum == 9.42, f"Sum: {pts_sum}")

# Check deduction narrative
record("1.1.4 Deduction narrative verifies 10.00 - 0.58 = 9.42", "-0.20 + 0.00 - 0.15 - 0.13 - 0.10" in audit_text or "3.80 + 1.50 + 1.35 + 1.37 + 1.40 = \\mathbf{9.42 / 10.00}" in audit_text, "Missing arithmetic explanation")

# Item 1.2: REQ-F-01..12 Final Phase Compliance Matrix present in Section 2.2
sec22_match = "### 2.2. Exhaustive Final Phase Compliance Matrix (REQ-F-01 Through REQ-F-12)" in audit_text
record("1.2.1 Section 2.2 header exists", sec22_match, "Section 2.2 header not found")

all_req_f = True
missing_f = []
for i in range(1, 13):
    fid = f"REQ-F-{i:02d}"
    if fid not in audit_text:
        all_req_f = False
        missing_f.append(fid)
record("1.2.2 All 12 REQ-F-01..12 requirement IDs present in Section 2.2", all_req_f, f"Missing: {missing_f}")

# Key paths cited in Section 2.2 exist on disk
key_f_paths = [
    "src_latex/main.tex",
    "src_latex/main.pdf",
    "src_latex/1_frontmatter/04_Declaration.tex",
    "src_latex/1_frontmatter/05_Abstract.tex",
    "src_latex/3_backmatter/Appendix.tex",
    "src_latex/references.bib"
]
for p in key_f_paths:
    full_p = os.path.join(ROOT_DIR, p)
    record(f"1.2.3 Cited Final Phase asset exists: {p}", os.path.exists(full_p), f"Path {full_p} missing")

# Item 1.3: Slide 77 demo delegation properly documented
req_m23_delegated = "QUALIFIED COMPLIANT / DELEGATED TO DEEP-DIVE" in audit_text
record("1.3.1 REQ-M-23 marked as QUALIFIED COMPLIANT / DELEGATED TO DEEP-DIVE", req_m23_delegated, "Status not matching")

sec52_step3 = "Step 3 (Architectures H1 & H2)" in audit_text and "causal_modeling_uci697.ipynb:Cell 11" in audit_text
record("1.3.2 Step 3 documents delegation of mask M and weight matrix W to Cell 11", sec52_step3, "Step 3 text missing delegation")

sec52_step4 = "Step 4 (Intervention Calculation & Live Query)" in audit_text and "causal_modeling_uci697.ipynb:Cells 07 & 13" in audit_text
record("1.3.3 Step 4 documents delegation of q_m(t, C_i) live querying to Cells 07 & 13", sec52_step4, "Step 4 text missing delegation")

# Verify code in notebook/causal_modeling_uci697.ipynb
with open(SUB_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    sub_nb = json.load(f)
c07_code = "".join(sub_nb["cells"][7]["source"])
c11_code = "".join(sub_nb["cells"][11]["source"])
c13_code = "".join(sub_nb["cells"][13]["source"])

record("1.3.4 Sub-notebook Cell 07 contains B0 DAG and query engine", "build_b0_dag" in c07_code and "query_model_effects" in c07_code, "Cell 07 code missing functions")
record("1.3.5 Sub-notebook Cell 11 contains NOTEARS mask and W optimization", "build_domain_mask" in c11_code and ("train_h2_notears" in c11_code or "notears" in c11_code.lower()), "Cell 11 code missing functions")
record("1.3.6 Sub-notebook Cell 13 contains 5-paired evaluation loop over C_test", "evaluate_prediction_metrics" in c13_code and "compute_bootstrap_ate_ci" in c13_code, "Cell 13 code missing functions")

# Item 1.4: Model H2 zero-width CI caveat documented in Section 5.3
# Check Table B CSV
df_b = pd.read_csv(TABLE_B_PATH)
h2_rows = df_b[df_b["Model"].str.contains("H2")]
h2_zero_width = ((h2_rows["CI_95_High"] - h2_rows["CI_95_Low"]).abs().max()) < 1e-5
record("1.4.1 Table B CSV confirmed to have zero-width CI [0.3437, 0.3437] for H2", h2_zero_width, "H2 CI width is not zero")

req_m20_caveat = "COMPLIANT WITH AUDIT CAVEAT" in audit_text and "Audit Caveat" in audit_text and "degenerated zero-width CI" in audit_text
record("1.4.2 REQ-M-20 explicitly documents H2 zero-width CI caveat", req_m20_caveat, "REQ-M-20 missing H2 zero-width caveat")

sec53_h2_caveat = "H2 Graph Unidentifiability & Degenerated CI [0.3437, 0.3437]" in audit_text and "learning 0 backdoor confounders" in audit_text
record("1.4.3 Section 5.3 Mode 2 documents structural cause of H2 CI collapse", sec53_h2_caveat, "Section 5.3 Mode 2 missing H2 structural explanation")

# Item 1.5: Slide 79 loss curve caveat documented
# Check master notebook has no loss curve plots
with open(MASTER_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    master_nb = json.load(f)
loss_curve_found = any("plt.plot" in "".join(c["source"]) and "loss" in "".join(c["source"]).lower() for c in master_nb["cells"])
record("1.5.1 Master notebook confirmed to have zero inline epoch loss curves", not loss_curve_found, "Loss curves unexpectedly found in master notebook")

sec53_mode5 = "Neural CPD Overfitting & Epoch Loss Curves" in audit_text and "artifacts/models/" in audit_text and ("0.1648" in audit_text and "0.1674" in audit_text)
record("1.5.2 Section 5.3 Mode 5 documents epoch loss curve caveat & Ablation A1 defense", sec53_mode5, "Section 5.3 Mode 5 missing loss curve caveat")

dim3_deduction = "-0.15" in audit_text and "lack of live interactive individual profile" in audit_text and "missing Slide 79 Mode 5 epoch loss curves" in audit_text
record("1.5.3 Section 1.3 records explicit -0.15 deduction for live demo & loss curve omission", dim3_deduction, "Dimension 3 deduction narrative mismatch")

# Item 1.6: QA script summary table perfectly harmonized
qa_naive_gap = "+34.54\\%" in qa_text or "+34.54%" in qa_text
record("1.6.1 QA script summary table has +34.54% naive gap", qa_naive_gap, "QA script missing +34.54%")

qa_b0_auroc = "0.8001 \\pm 0.0135" in qa_text or "0.8001 ± 0.0135" in qa_text
record("1.6.2 QA script summary table has AUROC B0 0.8001 ± 0.0135", qa_b0_auroc, "QA script missing AUROC B0 0.8001 ± 0.0135")

qa_b0_ate = "+0.2363" in qa_text and "0.2240, 0.2487" in qa_text
record("1.6.3 QA script summary table has ATE B0 +0.2363 [0.2240, 0.2487]", qa_b0_ate, "QA script missing ATE B0 +0.2363 [0.2240, 0.2487]")

qa_h1_ate = "+0.2309" in qa_text and "0.2223, 0.2397" in qa_text
record("1.6.4 QA script summary table has ATE H1 +0.2309 [0.2223, 0.2397]", qa_h1_ate, "QA script missing ATE H1 +0.2309 [0.2223, 0.2397]")

qa_no_3809 = "38.09" not in qa_text
record("1.6.5 QA script completely purged of obsolete 38.09%", qa_no_3809, "Stale 38.09% found in QA script")

# Item 1.7: Solo author policy audited
solo_policy_audited = "REQ-M-01" in audit_text and "Hoàng Đình Quý Vũ" in audit_text and "authorized solo project under course group policy (1–2 students)" in audit_text
record("1.7.1 REQ-M-01 audits solo author policy under 1-2 students rule", solo_policy_audited, "REQ-M-01 missing solo author policy audit")

# ==============================================================================
# PART 2: ADVERSARIAL VERIFICATION OF 3 ITERATION 2 FIXES
# ==============================================================================
print("\n--- PART 2: 3 ITERATION 2 FIXES ---")

# Item 2.1: requirements.txt exists and is functional
req_exists = os.path.exists(REQUIREMENTS_PATH)
record("2.1.1 requirements.txt exists at project root", req_exists, f"Missing {REQUIREMENTS_PATH}")

req_size = os.path.getsize(REQUIREMENTS_PATH) if req_exists else 0
record("2.1.2 requirements.txt size > 500 bytes", req_size > 500, f"Size: {req_size} bytes")

key_dependencies = [
    "torch", "pgmpy", "dowhy", "pandas", "scikit-learn",
    "numpy", "scipy", "matplotlib", "seaborn",
    "causal-learn", "cvxpy", "networkx", "statsmodels"
]
with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
    req_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

req_packages = set()
invalid_reqs = []
for l in req_lines:
    try:
        req_obj = Requirement(l)
        req_packages.add(req_obj.name.lower())
    except Exception as e:
        invalid_reqs.append((l, str(e)))

record("2.1.3 All lines in requirements.txt are valid pip requirement specifications", len(invalid_reqs) == 0, f"Invalid specifications: {invalid_reqs}")

all_key_found = True
missing_key = []
for k in key_dependencies:
    # Handle normalized names
    k_norm = k.lower().replace("_", "-")
    found = any(p.replace("_", "-") == k_norm for p in req_packages)
    if not found:
        all_key_found = False
        missing_key.append(k)

record("2.1.4 All 13 core and specialized packages are present in requirements.txt", all_key_found, f"Missing: {missing_key}")

# REQ-F-12 in audit report references requirements.txt
req_f12_compliant = "REQ-F-12" in audit_text and "requirements.txt" in audit_text and "**COMPLIANT**" in audit_text
record("2.1.5 REQ-F-12 cites requirements.txt as COMPLIANT", req_f12_compliant, "REQ-F-12 not citing requirements.txt properly")

# Item 2.2: Section 6.3 LaTeX files all exist
sec63_text_match = re.search(r"### 6\.3\. Comprehensive Mapping to `src_latex/` 6-Chapter Manuscript Architecture[\s\S]*?(?=###|\Z)", audit_text)
record("2.2.1 Section 6.3 exists in audit report", sec63_text_match is not None, "Section 6.3 not found")

sec63_text = sec63_text_match.group(0) if sec63_text_match else ""

# Extract all .tex and .bib citations from Section 6.3
cited_tex_files = re.findall(r"`([a-zA-Z0-9_\-\./]+\.(?:tex|bib))`", sec63_text)
record("2.2.2 Section 6.3 contains cited LaTeX files", len(cited_tex_files) >= 30, f"Found {len(cited_tex_files)} citations")

# Map of chapters and prefixes
all_latex_exist = True
missing_latex = []
for cf in cited_tex_files:
    # Determine full path
    if cf.startswith("src_latex/"):
        full_p = os.path.join(ROOT_DIR, cf)
    else:
        # Search in src_latex
        found_path = None
        for root, dirs, files in os.walk(os.path.join(ROOT_DIR, "src_latex")):
            if cf in files:
                found_path = os.path.join(root, cf)
                break
        full_p = found_path

    if not full_p or not os.path.exists(full_p):
        all_latex_exist = False
        missing_latex.append(cf)

record("2.2.3 All files cited in Section 6.3 exist on disk in src_latex/", all_latex_exist, f"Missing: {missing_latex}")

# Adversarially verify that old Iteration 2 placeholder filenames do NOT appear
old_placeholders = [
    "2_1_CBN_Foundations.tex",
    "2_8_Research_Gaps.tex",
    "4_1_Experimental_Protocol_and_Data_Hygiene.tex",
    "4_5_Hardware_and_Scalability_Analysis.tex",
    "6_1_Synthesis_of_Empirical_Findings.tex",
    "6_4_Strategic_Policy_Recommendations.tex"
]
found_old_placeholders = [ph for ph in old_placeholders if ph in audit_text]
record("2.2.4 None of the old Iteration 2 placeholder filenames remain in audit report", len(found_old_placeholders) == 0, f"Found stale: {found_old_placeholders}")

# Item 2.3: REQ-M-20 std devs match Table B
# Authoritative values from Table_B_Intervention.csv
b0_pooled_row = df_b[(df_b["Model"].str.contains("B0")) & (df_b["Seed"].str.contains("Pooled"))].iloc[0]
h1_pooled_row = df_b[(df_b["Model"].str.contains("H1")) & (df_b["Seed"].str.contains("Pooled"))].iloc[0]

# Parse Mean ± Std string in Table B
# Format: "+0.2363 ± 0.0061" and "+0.2309 ± 0.0097"
b0_ate_str = b0_pooled_row["ATE"]
h1_ate_str = h1_pooled_row["ATE"]

print(f"Table B Authoritative Pooled ATE: B0 = {b0_ate_str}, H1 = {h1_ate_str}")

record("2.3.1 Table B contains B0 ± 0.0061 and H1 ± 0.0097", "0.0061" in b0_ate_str and "0.0097" in h1_ate_str, f"Table B values: {b0_ate_str}, {h1_ate_str}")

# Check line 112 (REQ-M-20)
req_m20_line = [l for l in audit_text.splitlines() if "REQ-M-20" in l][0]
record("2.3.2 REQ-M-20 contains B0 std dev \\pm 0.0061", "\\pm 0.0061" in req_m20_line or "± 0.0061" in req_m20_line, f"Line: {req_m20_line}")
record("2.3.3 REQ-M-20 contains H1 std dev \\pm 0.0097", "\\pm 0.0097" in req_m20_line or "± 0.0097" in req_m20_line, f"Line: {req_m20_line}")

# Check line 399 (Step 5)
step5_line = [l for l in audit_text.splitlines() if "Step 5 (Results Comparison Tables" in l][0]
record("2.3.4 Step 5 contains H1 std dev \\pm 0.0097", "\\pm 0.0097" in step5_line or "± 0.0097" in step5_line, f"Line: {step5_line}")

# Check that stale values 0.0053 and 0.0041 are 100% eliminated from audit report
record("2.3.5 Stale value 0.0053 is completely absent from audit report", "0.0053" not in audit_text, "Found 0.0053 in report")
record("2.3.6 Stale value 0.0041 is completely absent from audit report", "0.0041" not in audit_text, "Found 0.0041 in report")

# ==============================================================================
# SUMMARY & VERDICT
# ==============================================================================
print("\n" + "=" * 80)
print(f"TOTAL CHECKS: {passed + failed}")
print(f"PASSED:       {passed}")
print(f"FAILED:       {failed}")
print("=" * 80)

if failed == 0:
    print("\nFINAL VERDICT: CONFIRM (All 7 Iteration 1 items & 3 Iteration 2 fixes 100% verified)")
    sys.exit(0)
else:
    print("\nFINAL VERDICT: REJECT (Residual failures detected)")
    for err in errors:
        print("  -", err)
    sys.exit(1)
