import os
import sys
import re
import json
import pandas as pd
import numpy as np

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = r"e:\01_Research_Study_Master\04_Probabilistic_GrModel"
os.chdir(WORKSPACE)

print("=" * 80)
print("EMPIRICAL CHALLENGER 1 (M1-IT2) AUDIT & VERIFICATION SUITE")
print("=" * 80)

results = {
    "passed": [],
    "failed": []
}

def record_check(name, passed, details=""):
    if passed:
        results["passed"].append((name, details))
        print(f"[PASS] {name}: {details}")
    else:
        results["failed"].append((name, details))
        print(f"[FAIL] {name}: {details}")

# ==============================================================================
# TEST 1: Arithmetic in Section 1.3
# ==============================================================================
print("\n--- 1. SECTION 1.3 ARITHMETIC VERIFICATION ---")
pts = [3.80, 1.50, 1.35, 1.37, 1.40]
exact_sum = round(sum(pts), 4)
record_check(
    "Arithmetic raw array sum == 9.42",
    exact_sum == 9.42,
    f"sum([3.80, 1.50, 1.35, 1.37, 1.40]) = {exact_sum}"
)

audit_path = "report/midterm/midterm_defense_audit_report.md"
with open(audit_path, "r", encoding="utf-8") as f:
    audit_text = f.read()

# Extract Section 1.3 table rows
sec_1_3_match = re.search(r"### 1\.3\. Readiness Score Assessment.*?```(.*?)```", audit_text, re.DOTALL)
if sec_1_3_match:
    table_content = sec_1_3_match.group(1)
    rows = [line for line in table_content.splitlines() if "│" in line and not "TOTAL" in line and not "Evaluation Dimension" in line and not "READINESS SCORE" in line]
    parsed_pts = []
    for r in rows:
        parts = [p.strip() for p in r.split("│") if p.strip()]
        if len(parts) >= 3:
            try:
                val = float(parts[2])
                parsed_pts.append(val)
            except ValueError:
                pass
    record_check(
        "Section 1.3 table parsed points match [3.80, 1.50, 1.35, 1.37, 1.40]",
        parsed_pts == pts,
        f"Parsed pts: {parsed_pts}"
    )
    parsed_sum = round(sum(parsed_pts), 4)
    record_check(
        "Section 1.3 table parsed sum equals 9.42",
        parsed_sum == 9.42,
        f"Parsed sum: {parsed_sum}"
    )
    record_check(
        "Section 1.3 reported total in table is 9.42 / 10",
        "9.42 / 10" in table_content,
        "Table contains '9.42 / 10'"
    )
else:
    record_check("Section 1.3 table found", False, "Could not find Section 1.3 table")


# ==============================================================================
# TEST 2: Every file path cited in Section 6.3 exists in src_latex/
# ==============================================================================
print("\n--- 2. SECTION 6.3 LATEX FILE PATH EXISTENCE ---")
sec_6_3_match = re.search(r"### 6\.3\. Comprehensive Mapping to `src_latex/` 6-Chapter Manuscript Architecture(.*?)(?=\n---\n|\n## 7\.|\Z)", audit_text, re.DOTALL)
if sec_6_3_match:
    sec_6_3_text = sec_6_3_match.group(1)
    print("Section 6.3 excerpt:\n", sec_6_3_text.strip())

    # Check Frontmatter
    frontmatter_files = [
        "01_Coverpage.tex", "02_Acknowledgements.tex", "03_CommitteeInfo.tex",
        "04_Declaration.tex", "05_Abstract.tex", "06_TableOfContents.tex",
        "07_ListOfFigures.tex", "08_ListOfTables.tex", "09_ListOfAbbreviations.tex"
    ]
    for fn in frontmatter_files:
        p = os.path.join("src_latex", "1_frontmatter", fn)
        record_check(f"Frontmatter: {fn}", os.path.exists(p), f"Path: {p}")

    # Check Chapter 1
    ch1_files = [
        "1_1_Reason_for_choosing_the_topic.tex",
        "1_2_Research_objectives.tex",
        "1_3_Research_questions_and_hypotheses.tex",
        "1_4_Research_subjects_and_scope.tex",
        "1_5_Contributions_of_the_study.tex",
        "1_6_Report_structure.tex"
    ]
    for fn in ch1_files:
        p = os.path.join("src_latex", "2_chapters", "1_Introduction", fn)
        record_check(f"Chapter 1: {fn}", os.path.exists(p), f"Path: {p}")

    # Check Chapter 2 cited files: 2_1_CBN_Foundations.tex to 2_8_Research_Gaps.tex
    ch2_cited_start = "2_1_CBN_Foundations.tex"
    ch2_cited_end = "2_8_Research_Gaps.tex"
    p_ch2_start = os.path.join("src_latex", "2_chapters", "2_Background", ch2_cited_start)
    p_ch2_end = os.path.join("src_latex", "2_chapters", "2_Background", ch2_cited_end)
    record_check(f"Chapter 2 cited start: {ch2_cited_start}", os.path.exists(p_ch2_start), f"Path: {p_ch2_start} exists={os.path.exists(p_ch2_start)}")
    record_check(f"Chapter 2 cited end: {ch2_cited_end}", os.path.exists(p_ch2_end), f"Path: {p_ch2_end} exists={os.path.exists(p_ch2_end)}")

    # Check Chapter 3
    ch3_files = [
        "3_1_Overall_Architecture_and_Domain_DAG.tex",
        "3_2_Factorial_Model_Matrix.tex",
        "3_3_Neural_CPD_and_Temporal_Mask.tex",
        "3_4_Vectorized_O1_Tensor_Engine.tex",
        "3_5_Refutation_Suite_and_Sensitivity.tex"
    ]
    for fn in ch3_files:
        p = os.path.join("src_latex", "2_chapters", "3_Methods_Models", fn)
        record_check(f"Chapter 3: {fn}", os.path.exists(p), f"Path: {p}")

    # Check Chapter 4 cited files: 4_1_Experimental_Protocol_and_Data_Hygiene.tex to 4_5_Hardware_and_Scalability_Analysis.tex
    ch4_cited_start = "4_1_Experimental_Protocol_and_Data_Hygiene.tex"
    ch4_cited_end = "4_5_Hardware_and_Scalability_Analysis.tex"
    p_ch4_start = os.path.join("src_latex", "2_chapters", "4_Experiments", ch4_cited_start)
    p_ch4_end = os.path.join("src_latex", "2_chapters", "4_Experiments", ch4_cited_end)
    record_check(f"Chapter 4 cited start: {ch4_cited_start}", os.path.exists(p_ch4_start), f"Path: {p_ch4_start} exists={os.path.exists(p_ch4_start)}")
    record_check(f"Chapter 4 cited end: {ch4_cited_end}", os.path.exists(p_ch4_end), f"Path: {p_ch4_end} exists={os.path.exists(p_ch4_end)}")

    # Check Chapter 5
    ch5_files = [
        "5_1_Alternative_Academic_Outcomes.tex",
        "5_2_Mediation_Analysis_Path_Decomposition.tex",
        "5_3_Heterogeneous_Treatment_Effects_CATE.tex",
        "5_4_Cinelli_Hazlett_Sensitivity_Contour.tex",
        "5_5_Budget_Constrained_Policy_Optimization.tex",
        "5_6_Large_Scale_Kaggle_Transferability_OOD.tex"
    ]
    for fn in ch5_files:
        p = os.path.join("src_latex", "2_chapters", "5_Advanced_Analysis", fn)
        record_check(f"Chapter 5: {fn}", os.path.exists(p), f"Path: {p}")

    # Check Chapter 6 cited files: 6_1_Synthesis_of_Empirical_Findings.tex to 6_4_Strategic_Policy_Recommendations.tex
    ch6_cited_start = "6_1_Synthesis_of_Empirical_Findings.tex"
    ch6_cited_end = "6_4_Strategic_Policy_Recommendations.tex"
    p_ch6_start = os.path.join("src_latex", "2_chapters", "6_Conclusion", ch6_cited_start)
    p_ch6_end = os.path.join("src_latex", "2_chapters", "6_Conclusion", ch6_cited_end)
    record_check(f"Chapter 6 cited start: {ch6_cited_start}", os.path.exists(p_ch6_start), f"Path: {p_ch6_start} exists={os.path.exists(p_ch6_start)}")
    record_check(f"Chapter 6 cited end: {ch6_cited_end}", os.path.exists(p_ch6_end), f"Path: {p_ch6_end} exists={os.path.exists(p_ch6_end)}")

    # Check Backmatter
    backmatter_files = [
        "Appendix.tex", "Publications.tex", "References.tex"
    ]
    for fn in backmatter_files:
        p = os.path.join("src_latex", "3_backmatter", fn)
        record_check(f"Backmatter: {fn}", os.path.exists(p), f"Path: {p}")
    p_bib = os.path.join("src_latex", "references.bib")
    record_check("references.bib", os.path.exists(p_bib), f"Path: {p_bib}")
else:
    record_check("Section 6.3 found", False, "Could not find Section 6.3 in audit report")


# ==============================================================================
# TEST 3: REQ-F-01 through REQ-F-12 are all present in Section 2.2
# ==============================================================================
print("\n--- 3. SECTION 2.2 REQ-F-01 THROUGH REQ-F-12 PRESENCE ---")
sec_2_2_match = re.search(r"### 2\.2\. Exhaustive Final Phase Compliance Matrix.*?(?=\n---\n|\n## 3\.)", audit_text, re.DOTALL)
if sec_2_2_match:
    sec_2_2_text = sec_2_2_match.group(0)
    for i in range(1, 13):
        f_id = f"REQ-F-{i:02d}"
        present = f_id in sec_2_2_text
        record_check(f"Presence of {f_id} in Section 2.2", present, f"Found: {present}")
else:
    record_check("Section 2.2 found", False, "Could not find Section 2.2 in audit report")


# ==============================================================================
# TEST 4: NOTEBOOK_DEFENSE_QA_SCRIPT.md Summary Table and Discrepancies
# ==============================================================================
print("\n--- 4. NOTEBOOK_DEFENSE_QA_SCRIPT.md SUMMARY TABLE & VALUE AUDIT ---")
qa_path = "report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md"
with open(qa_path, "r", encoding="utf-8") as f:
    qa_text = f.read()

# Check that +38.09% is nowhere present
record_check(
    "Absence of '+38.09%' in entire QA script",
    "+38.09%" not in qa_text and "38.09%" not in qa_text,
    f"'38.09%' in QA script: {'38.09%' in qa_text}"
)

# Extract Summary Table
table_qa_match = re.search(r"## BẢNG TỔNG HỢP CÁC CON SỐ CỐT LÕI CẦN THUỘC LÒNG KHI BẢO VỆ.*?(?=\n---\n|\Z)", qa_text, re.DOTALL)
if table_qa_match:
    table_qa_text = table_qa_match.group(0)

    # Check naive gap is +34.54%
    naive_match = "34.54" in table_qa_text and "86.17" in table_qa_text and "51.63" in table_qa_text
    record_check("QA Table: naive gap is +34.54% (86.17% - 51.63%)", naive_match, f"Found: {naive_match}")

    # Check AUROC B0 is 0.8001 +/- 0.0135
    auroc_b0_match = "0.8001" in table_qa_text and "0.0135" in table_qa_text
    record_check("QA Table: AUROC B0 is 0.8001 +/- 0.0135", auroc_b0_match, f"Found: {auroc_b0_match}")

    # Check ATE B0 is +0.2363 [0.2240, 0.2487]
    ate_b0_match = "+0.2363" in table_qa_text and "0.2240" in table_qa_text and "0.2487" in table_qa_text
    record_check("QA Table: ATE B0 is +0.2363 [0.2240, 0.2487]", ate_b0_match, f"Found: {ate_b0_match}")

    # Check ATE H1 is +0.2309 [0.2223, 0.2397]
    ate_h1_match = "+0.2309" in table_qa_text and "0.2223" in table_qa_text and "0.2397" in table_qa_text
    record_check("QA Table: ATE H1 is +0.2309 [0.2223, 0.2397]", ate_h1_match, f"Found: {ate_h1_match}")
else:
    record_check("QA Summary Table found", False, "Could not find Summary Table in QA script")


# ==============================================================================
# TEST 5: docs/code_instruction/ parent directory prefix in REQ-M-03/07/08/09
# ==============================================================================
print("\n--- 5. docs/code_instruction/ PREFIX IN REQ-M-03/07/08/09 ---")
for req_id in ["REQ-M-03", "REQ-M-07", "REQ-M-08", "REQ-M-09"]:
    pattern = rf"\|\s*\*\*{req_id}\*\*.*?\n"
    match = re.search(pattern, audit_text)
    if match:
        line_str = match.group(0)
        has_prefix = "docs/code_instruction/" in line_str
        record_check(f"Prefix 'docs/code_instruction/' in {req_id}", has_prefix, f"Line snippet: {line_str.strip()[:120]}")
    else:
        record_check(f"Row {req_id} found in Section 2.1", False, "Row not found")


# ==============================================================================
# TEST 6: Deep-dive notebook cell indices for notebook/causal_modeling_uci697.ipynb
# ==============================================================================
print("\n--- 6. DEEP-DIVE NOTEBOOK CELL INDICES VERIFICATION ---")
nb_path = "notebook/causal_modeling_uci697.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb_data = json.load(f)

cells = nb_data.get("cells", [])
print(f"Total cells in {nb_path}: {len(cells)}")

# Cell 07 (B0)
c7 = cells[7]
c7_src = "".join(c7.get("source", []))
c7_is_b0 = "build_b0_dag" in c7_src and "fit_b0_model" in c7_src and "query_model_effects" in c7_src
record_check(
    "Cell 07 is Model B0 construction & fitting",
    c7_is_b0,
    f"Cell 07 type={c7.get('cell_type')}, header={c7_src.splitlines()[1] if len(c7_src.splitlines()) > 1 else 'None'}"
)

# Cell 09 (H1 MLP)
c9 = cells[9]
c9_src = "".join(c9.get("source", []))
c9_is_h1 = "NeuralCPDModule" in c9_src and "train_h1_mlp" in c9_src and "build_h1_bayesian_network" in c9_src
record_check(
    "Cell 09 is Model H1 PyTorch MLP & Neural CPD",
    c9_is_h1,
    f"Cell 09 type={c9.get('cell_type')}, header={c9_src.splitlines()[1] if len(c9_src.splitlines()) > 1 else 'None'}"
)

# Cell 13 (5-paired standardized ATE loop)
c13 = cells[13]
c13_src = "".join(c13.get("source", []))
c13_is_ate = "evaluate_prediction_metrics" in c13_src and "compute_bootstrap_ate_ci" in c13_src and "5 Paired Runs" in c13_src
record_check(
    "Cell 13 is 5-paired standardized ATE evaluation loop",
    c13_is_ate,
    f"Cell 13 type={c13.get('cell_type')}, header={c13_src.splitlines()[1] if len(c13_src.splitlines()) > 1 else 'None'}"
)

# Also check citations in audit report and QA script
record_check(
    "Audit report cites causal_modeling_uci697.ipynb:Cell 07 for B0",
    "causal_modeling_uci697.ipynb:Cell 07" in audit_text,
    "Cited in REQ-M-07 / Section 1.4"
)
record_check(
    "Audit report cites causal_modeling_uci697.ipynb:Cell 09 for H1",
    "causal_modeling_uci697.ipynb:Cell 09" in audit_text,
    "Cited in REQ-M-08 / Section 1.4"
)
record_check(
    "Audit report cites causal_modeling_uci697.ipynb:Cell 13 for 5-paired ATE",
    "causal_modeling_uci697.ipynb:Cell 13" in audit_text,
    "Cited in REQ-M-13 / Section 1.4"
)


# ==============================================================================
# SUMMARY & VERDICT CALCULATION
# ==============================================================================
print("\n" + "=" * 80)
print(f"VERIFICATION RESULTS: {len(results['passed'])} PASSED, {len(results['failed'])} FAILED")
print("=" * 80)
if results["failed"]:
    print("\nFAILURES DETECTED:")
    for name, det in results["failed"]:
        print(f"  - {name}: {det}")
else:
    print("\nALL EMPIRICAL CHECKS PASSED PERFECTLY!")

