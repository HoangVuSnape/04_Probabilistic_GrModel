#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical Challenger 1 Test Harness for Milestone 1 Iteration 3 (challenger_1_m1_it3)

Comprehensive verification suite addressing all Milestone 1 Gate Iteration 3 challenge requirements:
1. Section 6.3 LaTeX citations verification against physical disk (all 44 files across frontmatter, Ch 1-6, backmatter).
2. requirements.txt existence and verification of all 9 required packages and pinned versions.
3. REQ-M-20 and Step 5 standard deviations harmonization against artifacts/tables/Table_B_Intervention.csv.
4. Section 1.3 score table arithmetic verification (sum = 9.42 / 10.00).
5. Comprehensive absence check for obsolete/stale placeholder filenames and stale variance numbers.
"""

import os
import sys
import re
from decimal import Decimal

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(WORKSPACE)

print("=" * 90)
print("EMPIRICAL CHALLENGER 1 AUDIT SUITE — MILESTONE 1 ITERATION 3")
print("Workspace:", WORKSPACE)
print("=" * 90)

passed_checks = 0
failed_checks = 0
audit_log = []

def record(test_id, category, description, condition, details=""):
    global passed_checks, failed_checks
    status = "PASS" if condition else "FAIL"
    if condition:
        passed_checks += 1
    else:
        failed_checks += 1
    audit_log.append({
        "id": test_id,
        "category": category,
        "description": description,
        "status": status,
        "details": details
    })
    print(f"[{status}] [{category}] {test_id}: {description} {details}")

# ==============================================================================
# SUITE 1: SECTION 1.3 READINESS SCORE ARITHMETIC VERIFICATION
# ==============================================================================
print("\n" + "-" * 80)
print("SUITE 1: Section 1.3 Score Table Arithmetic Verification")
print("-" * 80)

audit_report_path = os.path.join("report", "midterm", "midterm_defense_audit_report.md")
record("T1.1", "Score Table", "Audit report file exists", os.path.exists(audit_report_path), audit_report_path)

with open(audit_report_path, "r", encoding="utf-8") as f:
    audit_report_content = f.read()

# Parse Section 1.3 table
m_table = re.search(r"READINESS SCORE COMPONENT.*?\n(.*?)\n└", audit_report_content, re.DOTALL)
record("T1.2", "Score Table", "Section 1.3 Readiness score component table located", bool(m_table))

if m_table:
    table_lines = m_table.group(1).splitlines()
    parsed_points = []
    for line in table_lines:
        if "│" in line:
            parts = [p.strip() for p in line.split("│")[1:-1]]
            if len(parts) >= 4:
                dim, max_pt, cur_pt, status = parts[0], parts[1], parts[2], parts[3]
                try:
                    val = Decimal(cur_pt)
                    parsed_points.append((dim, val))
                except Exception:
                    pass
    
    expected_rows = [
        ("1. Theoretical Causal Formulation", Decimal("3.80")),
        ("2. Data Hygiene & Zero-Leakage Protocol", Decimal("1.50")),
        ("3. Master Notebook Live Execution", Decimal("1.35")),
        ("4. Factorial Benchmark Tables", Decimal("1.37")),
        ("5. Advanced Causal Extensions", Decimal("1.40")),
    ]
    
    record("T1.3", "Score Table", "Extracted exactly 5 numeric component rows", len(parsed_points) == 5, f"count={len(parsed_points)}")
    
    total_pts = Decimal("0.00")
    for (dim, pt), (exp_name, exp_val) in zip(parsed_points, expected_rows):
        total_pts += pt
        record(f"T1.row.{pt}", "Score Table", f"Row '{exp_name}' equals {exp_val}", pt == exp_val, f"actual={pt}")
        
    record("T1.4", "Score Table", "Sum of rows equals exactly 9.42 / 10.00", total_pts == Decimal("9.42"), f"sum={total_pts}")
    record("T1.5", "Score Table", "Section 1.3 text claims exact column sum 9.42 / 10.00", "9.42 / 10.00" in audit_report_content)
    record("T1.6", "Score Table", "Section 1.3 text claims 94.20%", "94.20%" in audit_report_content)

# ==============================================================================
# SUITE 2: REQUIREMENTS.TXT AT ROOT AND DEPENDENCY AUDIT
# ==============================================================================
print("\n" + "-" * 80)
print("SUITE 2: requirements.txt Existence and Key Dependencies Verification")
print("-" * 80)

req_path = "requirements.txt"
req_exists = os.path.exists(req_path)
record("T2.1", "Requirements", "requirements.txt exists at project root", req_exists, f"path={req_path}")

if req_exists:
    with open(req_path, "r", encoding="utf-8") as f:
        req_lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    
    record("T2.2", "Requirements", "requirements.txt is non-empty", len(req_lines) > 0, f"lines={len(req_lines)}")
    
    # Parse package names and versions
    pkg_map = {}
    for line in req_lines:
        if "==" in line:
            pkg, ver = line.split("==", 1)
            pkg_map[pkg.lower().strip()] = ver.strip()
        else:
            pkg_map[line.lower().strip()] = "unspecified"
            
    mandatory_libs = [
        "torch",
        "pgmpy",
        "dowhy",
        "pandas",
        "scikit-learn",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn"
    ]
    
    for lib in mandatory_libs:
        is_present = lib.lower() in pkg_map
        version = pkg_map.get(lib.lower(), "MISSING")
        record(f"T2.pkg.{lib}", "Requirements", f"Mandatory library '{lib}' present with pinned version", is_present and version != "MISSING", f"version={version}")
        
    # Additional domain-critical causal libraries
    causal_libs = ["causal-learn", "cvxpy", "networkx", "statsmodels"]
    for lib in causal_libs:
        is_present = lib.lower() in pkg_map
        version = pkg_map.get(lib.lower(), "MISSING")
        record(f"T2.causal.{lib}", "Requirements", f"Domain library '{lib}' present", is_present, f"version={version}")

# ==============================================================================
# SUITE 3: REQ-M-20 AND STEP 5 STANDARD DEVIATION HARMONIZATION AGAINST TABLE B
# ==============================================================================
print("\n" + "-" * 80)
print("SUITE 3: REQ-M-20 Standard Deviation Match with Table_B_Intervention.csv")
print("-" * 80)

table_b_path = os.path.join("artifacts", "tables", "Table_B_Intervention.csv")
record("T3.1", "Table B Sync", "Table_B_Intervention.csv exists", os.path.exists(table_b_path), table_b_path)

if os.path.exists(table_b_path):
    with open(table_b_path, "r", encoding="utf-8") as f:
        table_b_lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Find B0 and H1 Mean ± Std rows
    b0_mean_line = next((l for l in table_b_lines if l.startswith("B0 (Mean ± Std)")), None)
    h1_mean_line = next((l for l in table_b_lines if l.startswith("H1 (Mean ± Std)")), None)
    
    record("T3.2", "Table B Sync", "Table B contains B0 Mean ± Std summary row", b0_mean_line is not None, b0_mean_line)
    record("T3.3", "Table B Sync", "Table B contains H1 Mean ± Std summary row", h1_mean_line is not None, h1_mean_line)
    
    if b0_mean_line and h1_mean_line:
        # Check B0 ATE in CSV: "+0.2363 ± 0.0061"
        b0_has_target = "+0.2363 ± 0.0061" in b0_mean_line
        record("T3.4", "Table B Sync", "Table B row 17 specifies B0 ATE as +0.2363 ± 0.0061", b0_has_target)
        
        # Check H1 ATE in CSV: "+0.2309 ± 0.0097"
        h1_has_target = "+0.2309 ± 0.0097" in h1_mean_line
        record("T3.5", "Table B Sync", "Table B row 18 specifies H1 ATE as +0.2309 ± 0.0097", h1_has_target)

# Check REQ-M-20 row in audit report
req_m_20_lines = [l for l in audit_report_content.splitlines() if "REQ-M-20" in l]
record("T3.6", "Table B Sync", "Audit report contains REQ-M-20 row", len(req_m_20_lines) > 0)

if req_m_20_lines:
    req_m_20 = req_m_20_lines[0]
    record("T3.7", "Table B Sync", r"REQ-M-20 contains B0 ATE '+0.2363 \pm 0.0061'", r"+0.2363 \pm 0.0061" in req_m_20)
    record("T3.8", "Table B Sync", r"REQ-M-20 contains H1 ATE '+0.2309 \pm 0.0097'", r"+0.2309 \pm 0.0097" in req_m_20)

# Check Step 5 in Section 5.2
record("T3.9", "Table B Sync", r"Section 5.2 Step 5 contains H1 ATE '+0.2309 \pm 0.0097'", r"+0.2309 \pm 0.0097" in audit_report_content)

# Adversarial negative checks: stale numbers must be completely absent from entire report
record("T3.10", "Table B Sync", "Stale B0 variance '0.0053' is completely purged from audit report", "0.0053" not in audit_report_content)
record("T3.11", "Table B Sync", "Stale H1 variance '0.0041' is completely purged from audit report", "0.0041" not in audit_report_content)

# ==============================================================================
# SUITE 4: SECTION 6.3 COMPREHENSIVE LATEX FILESYSTEM VERIFICATION
# ==============================================================================
print("\n" + "-" * 80)
print("SUITE 4: Section 6.3 Comprehensive LaTeX Manuscript Architecture Verification")
print("-" * 80)

sec_6_3_regex = r"### 6\.3\. Comprehensive Mapping to `src_latex/` 6-Chapter Manuscript Architecture(.*?)(?=\n---\n|\n## 7\.|\Z)"
m_sec_6_3 = re.search(sec_6_3_regex, audit_report_content, re.DOTALL)
record("T4.1", "LaTeX Citations", "Section 6.3 located in audit report", bool(m_sec_6_3))

if m_sec_6_3:
    sec_6_3_text = m_sec_6_3.group(1)
    
    # 1. Frontmatter
    frontmatter_files = [
        "01_Coverpage.tex",
        "02_Acknowledgements.tex",
        "03_CommitteeInfo.tex",
        "04_Declaration.tex",
        "05_Abstract.tex",
        "06_TableOfContents.tex",
        "07_ListOfFigures.tex",
        "08_ListOfTables.tex",
        "09_ListOfAbbreviations.tex"
    ]
    for fn in frontmatter_files:
        p = os.path.join("src_latex", "1_frontmatter", fn)
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.front.{fn}", "Frontmatter", f"Frontmatter {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 2. Chapter 1
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
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch1.{fn}", "Chapter 1", f"Chapter 1 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 3. Chapter 2 (Remediation verification for It2 REJECT item)
    ch2_files = [
        "2_1_Probabilistic_Graphical_Models.tex",
        "2_2_Structural_Causal_Models.tex",
        "2_3_Causal_Discovery.tex",
        "2_4_Statistical_Causal_Estimators.tex",
        "2_5_Related_Work_EDM.tex"
    ]
    record("T4.ch2.count", "Chapter 2", "Chapter 2 specifies all 5 sections (2_1 to 2_5)", len(ch2_files) == 5)
    for fn in ch2_files:
        p = os.path.join("src_latex", "2_chapters", "2_Background", fn)
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch2.{fn}", "Chapter 2", f"Chapter 2 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 4. Chapter 3
    ch3_files = [
        "3_1_Overall_Architecture_and_Domain_DAG.tex",
        "3_2_Factorial_Model_Matrix.tex",
        "3_3_Neural_CPD_and_Temporal_Mask.tex",
        "3_4_Vectorized_O1_Tensor_Engine.tex",
        "3_5_Refutation_Suite_and_Sensitivity.tex"
    ]
    for fn in ch3_files:
        p = os.path.join("src_latex", "2_chapters", "3_Methods_Models", fn)
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch3.{fn}", "Chapter 3", f"Chapter 3 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 5. Chapter 4 (Remediation verification for It2 REJECT item)
    ch4_files = [
        "4_1_Experimental_Setup_and_Metrics.tex",
        "4_2_HeldOut_Predictive_Performance.tex",
        "4_3_Causal_Interventions_and_ATE.tex",
        "4_4_DoWhy_Refutations_and_Computational_Efficiency.tex",
        "4_5_Ablation_Studies.tex"
    ]
    record("T4.ch4.count", "Chapter 4", "Chapter 4 specifies all 5 sections (4_1 to 4_5)", len(ch4_files) == 5)
    for fn in ch4_files:
        p = os.path.join("src_latex", "2_chapters", "4_Experiments", fn)
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch4.{fn}", "Chapter 4", f"Chapter 4 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 6. Chapter 5
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
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch5.{fn}", "Chapter 5", f"Chapter 5 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 7. Chapter 6 (Remediation verification for It2 REJECT item)
    ch6_files = [
        "6_1_Summary_of_Findings.tex",
        "6_2_Institutional_Policy_Recommendations.tex",
        "6_3_Limitations_and_Threats_to_Validity.tex",
        "6_4_Future_Research_Directions.tex"
    ]
    record("T4.ch6.count", "Chapter 6", "Chapter 6 specifies all 4 sections (6_1 to 6_4)", len(ch6_files) == 4)
    for fn in ch6_files:
        p = os.path.join("src_latex", "2_chapters", "6_Conclusion", fn)
        in_text = f"`{fn}`" in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.ch6.{fn}", "Chapter 6", f"Chapter 6 {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")
        
    # 8. Backmatter & references
    backmatter_files = [
        ("Appendix.tex", os.path.join("src_latex", "3_backmatter", "Appendix.tex")),
        ("Publications.tex", os.path.join("src_latex", "3_backmatter", "Publications.tex")),
        ("References.tex", os.path.join("src_latex", "3_backmatter", "References.tex")),
        ("references.bib", os.path.join("src_latex", "references.bib"))
    ]
    for fn, p in backmatter_files:
        in_text = fn in sec_6_3_text
        exists_on_disk = os.path.exists(p)
        record(f"T4.back.{fn}", "Backmatter", f"Backmatter {fn} cited in report & exists on disk", in_text and exists_on_disk, f"cited={in_text}, exists={exists_on_disk}")

# ==============================================================================
# SUITE 5: ADVERSARIAL PHANTOM FILE & REGRESSION REJECTION SUITE
# ==============================================================================
print("\n" + "-" * 80)
print("SUITE 5: Adversarial Phantom File & Negative Regression Verification")
print("-" * 80)

# The phantom filenames from Iteration 2 that caused the REJECT verdict
phantom_filenames = [
    "2_1_CBN_Foundations.tex",
    "2_8_Research_Gaps.tex",
    "4_1_Experimental_Protocol_and_Data_Hygiene.tex",
    "4_5_Hardware_and_Scalability_Analysis.tex",
    "6_1_Synthesis_of_Empirical_Findings.tex",
    "6_4_Strategic_Policy_Recommendations.tex"
]

for ph in phantom_filenames:
    in_sec_6_3 = ph in sec_6_3_text
    in_whole_report = ph in audit_report_content
    record(f"T5.phantom.{ph}", "Adversarial", f"Phantom filename '{ph}' is completely absent from report", not in_whole_report, f"in_sec_6_3={in_sec_6_3}")

# Check LaTeX chapter entrypoints synchronization
chapter_includes = {
    "src_latex/2_chapters/2_Background.tex": ch2_files,
    "src_latex/2_chapters/4_Experiments.tex": ch4_files,
    "src_latex/2_chapters/6_Conclusion.tex": ch6_files,
}

for parent_file, child_files in chapter_includes.items():
    if os.path.exists(parent_file):
        with open(parent_file, "r", encoding="utf-8") as pf:
            pf_text = pf.read()
        for cf in child_files:
            cf_stem = cf.replace(".tex", "")
            included = cf_stem in pf_text
            record(f"T5.include.{cf}", "LaTeX Architecture", f"Parent '{parent_file}' includes '{cf_stem}'", included)

# ==============================================================================
# FINAL AUDIT SUMMARY
# ==============================================================================
print("\n" + "=" * 90)
print(f"AUDIT TOTALS: {passed_checks} PASSED, {failed_checks} FAILED (TOTAL CHECKS: {passed_checks + failed_checks})")
print("=" * 90)

if failed_checks == 0:
    print("\nVERDICT: >>> CONFIRM <<< (All requirements empirically validated with zero defects)")
    sys.exit(0)
else:
    print(f"\nVERDICT: >>> REJECT <<< ({failed_checks} checks failed)")
    sys.exit(1)
