import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== CHECKING CITATIONS IN midterm_defense_audit_report.md ===")

def check_citation(filepath, start_line, end_line, expected_snippet=""):
    if not os.path.exists(filepath):
        print(f"[MISSING FILE] {filepath}")
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total_lines = len(lines)
    if start_line > total_lines:
        print(f"[LINE OUT OF RANGE] {filepath}:{start_line}-{end_line} (total lines: {total_lines})")
        return False
    actual_snippet = "".join(lines[start_line-1:min(end_line, total_lines)]).strip()
    match = expected_snippet.lower() in actual_snippet.lower() if expected_snippet else True
    print(f"[{'PASS' if match else 'CONTENT MISMATCH'}] {filepath}:{start_line}-{end_line}")
    if not match:
        print(f"   Expected: {expected_snippet}")
        print(f"   Actual snippet:\n{actual_snippet[:150]}...")
    return match

# 1. docs/requirements/README.md:1-8
check_citation("docs/requirements/README.md", 1, 8, "IT701100")

# 2. docs/requirements/midterm/README.md:1-6
check_citation("docs/requirements/midterm/README.md", 1, 6, "Accept / Reject")

# 3. docs/requirements/midterm/README.md:7-9
check_citation("docs/requirements/midterm/README.md", 7, 9, "1.0")

# 4. docs/requirements/midterm/README.md:48-105
check_citation("docs/requirements/midterm/README.md", 48, 105, "Rubric")

# 5. report/midterm/midterm_report.md:4
check_citation("report/midterm/midterm_report.md", 4, 4)

# 6. report/midterm/midterm_report.md:32-38
check_citation("report/midterm/midterm_report.md", 32, 38)

# 7. report/midterm/midterm_report.md:44-58
check_citation("report/midterm/midterm_report.md", 44, 58)

# 8. report/midterm/midterm_report.md:60-72
check_citation("report/midterm/midterm_report.md", 60, 72)

# 9. report/midterm/midterm_report.md:82-84
check_citation("report/midterm/midterm_report.md", 82, 84)

# 10. report/midterm/midterm_report.md:85-89
check_citation("report/midterm/midterm_report.md", 85, 89)

# 11. report/midterm/midterm_report.md:92-102
check_citation("report/midterm/midterm_report.md", 92, 102)

# 12. report/midterm/midterm_report.md:103-110
check_citation("report/midterm/midterm_report.md", 103, 110)

# 13. report/midterm/midterm_report.md:120-129
check_citation("report/midterm/midterm_report.md", 120, 129)

# 14. report/midterm/midterm_report.md:130-136
check_citation("report/midterm/midterm_report.md", 130, 136)

# 15. report/midterm/midterm_report.md:144-158
check_citation("report/midterm/midterm_report.md", 144, 158)

# 16. report/midterm/midterm_report.md:230-245
check_citation("report/midterm/midterm_report.md", 230, 245)

# 17. 00_scope_and_contract.md vs docs/code_instruction/00_scope_and_contract.md
print("\n--- Check 00_scope_and_contract.md citation ---")
check_citation("00_scope_and_contract.md", 42, 51)
check_citation("docs/code_instruction/00_scope_and_contract.md", 42, 51)

# 18. 02_models_and_ablations.md vs docs/code_instruction/02_models_and_ablations.md
print("\n--- Check 02_models_and_ablations.md citation ---")
check_citation("02_models_and_ablations.md", 42, 47)
check_citation("docs/code_instruction/02_models_and_ablations.md", 42, 47)
check_citation("docs/code_instruction/02_models_and_ablations.md", 54, 64)
check_citation("docs/code_instruction/02_models_and_ablations.md", 67, 78)

# 19. tools/verify_and_compute_multi_treatments.py:77
print("\n--- Check tools citations ---")
check_citation("tools/verify_and_compute_multi_treatments.py", 70, 85)

# 20. tools/build_causal_notebook.py:108 & 194
check_citation("tools/build_causal_notebook.py", 100, 115)
check_citation("tools/build_causal_notebook.py", 190, 200)

# 21. notebook/causal_modeling_uci697.ipynb cells
print("\n--- Check notebook/causal_modeling_uci697.ipynb cells ---")
import json
with open("notebook/causal_modeling_uci697.ipynb", "r", encoding="utf-8") as f:
    nb_cm = json.load(f)
print(f"Total cells in causal_modeling_uci697.ipynb: {len(nb_cm['cells'])}")
for idx in [5, 6, 7, 11]:
    if idx < len(nb_cm['cells']):
        c = nb_cm['cells'][idx]
        src = "".join(c.get('source', []))[:100].replace("\n", " ")
        print(f"Cell {idx} ({c['cell_type']}): {src}...")
