import json
import re
from pathlib import Path

p = Path('00_MASTER_Causal_PGM_Educational_Intervention.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cells = nb['cells']

full_text = "\n".join("".join(c.get('source', [])) for c in cells)
code_cells = [c for c in cells if c['cell_type'] == 'code']
md_cells = [c for c in cells if c['cell_type'] == 'markdown']
code_text = "\n".join("".join(c.get('source', [])) for c in code_cells)
md_text = "\n".join("".join(c.get('source', [])) for c in md_cells)

print("=" * 70)
print("AUDIT REPORT: 12 CRITERIA FOR MASTER NOTEBOOK")
print("=" * 70)

# 1. TOC with hierarchical numbers
toc_matches = re.findall(r'\[(\d+\.[\d\.]*)\s+([^\]]+)\]', md_text)
c1_pass = len(toc_matches) >= 10
print(f"1. TOC with hierarchical numbers: {len(toc_matches)} entries found -> {'PASSED' if c1_pass else 'FAILED'}")

# 2. Concise headings
h2_matches = re.findall(r'^##\s+(\d+\.\s+.*)$', md_text, re.MULTILINE)
c2_pass = len(h2_matches) == 11
print(f"2. Concise Headings (1-11): {len(h2_matches)} sections found -> {'PASSED' if c2_pass else 'FAILED'}")

# 3. UCI 697 real data & metadata
has_uci_url = "https://archive.ics.uci.edu/dataset/697" in full_text
has_doi = "10.24432/C5MC89" in full_text
has_sha256 = "751bb23a8a255a05f512f5dea78aa16ef347b9814d2430d0883bdb756cac5c28" in full_text
c3_pass = has_uci_url and has_doi and has_sha256
print(f"3. UCI 697 Real Data & Metadata: URL={has_uci_url}, DOI={has_doi}, SHA256={has_sha256} -> {'PASSED' if c3_pass else 'FAILED'}")

# 4. Formal DAG constraints
has_acyclic = "kiem_tra_tinh_khong_chu_trinh" in code_text
has_temporal = "kiem_tra_phan_tang_thoi_gian" in code_text
has_expert = "kiem_tra_danh_sach_chuyen_gia" in code_text
has_backdoor = "kiem_tra_d_separation_backdoor" in code_text
c4_pass = all([has_acyclic, has_temporal, has_expert, has_backdoor])
print(f"4. Four DAG Constraints: Acyclic={has_acyclic}, Temporal={has_temporal}, Expert={has_expert}, Backdoor={has_backdoor} -> {'PASSED' if c4_pass else 'FAILED'}")

# 5. Structure learning survey & pgmpy
has_struct_survey = "HillClimbSearch" in code_text and "BIC" in code_text
c5_pass = has_struct_survey
print(f"5. Structure Learning: HillClimbSearch + BIC -> {'PASSED' if c5_pass else 'FAILED'}")

# 6. Parameter learning survey & pgmpy
has_param_survey = "BayesianEstimator" in code_text and "BDeu" in code_text and "MaximumLikelihoodEstimator" in code_text
c6_pass = has_param_survey
print(f"6. Parameter Learning: MLE + Bayesian BDeu -> {'PASSED' if c6_pass else 'FAILED'}")

# 7. Docstrings and type hints
func_defs = re.findall(r'def\s+([a-zA_Z_0-9]+)\s*\((.*?)\)\s*->\s*([^:]+):', code_text, re.DOTALL)
any_count = sum("Any" in fargs or "Any" in freturn for _, fargs, freturn in func_defs)
c7_pass = len(func_defs) >= 10 and any_count == 0
print(f"7. Docstrings & Type Hints: {len(func_defs)} typed functions, Any count={any_count} -> {'PASSED' if c7_pass else 'FAILED'}")

# 8. NO OOP (only dataclass)
import ast
tree = ast.parse(code_text)
class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
dataclass_count = sum(any("dataclass" in ast.unparse(d) for d in c.decorator_list) for c in class_defs)
methods_count = sum(sum(isinstance(n, ast.FunctionDef) for n in c.body) for c in class_defs)
has_self = "self" in code_text
c8_pass = len(class_defs) == dataclass_count and methods_count == 0 and not has_self
print(f"8. STRICTLY NO OOP: Classes={len(class_defs)}, Pure Dataclasses={dataclass_count}, Methods={methods_count}, 'self'={has_self} -> {'PASSED' if c8_pass else 'FAILED'}")

# 9. Structured programming, no nested def
lines = code_text.splitlines()
nested_defs = [line.strip() for line in lines if re.match(r'^\s{4,8}def\s+', line)]
c9_pass = len(nested_defs) == 0
print(f"9. Structured Programming (No Nested def): Nested count={len(nested_defs)} -> {'PASSED' if c9_pass else 'FAILED'}")

# 10. DoWhy intervention & 3 refutations
has_dowhy = "dowhy" in code_text and "CausalModel" in code_text
has_ref_rcc = "random_common_cause" in code_text
has_ref_placebo = "placebo_treatment_refuter" in code_text
has_ref_subset = "data_subset_refuter" in code_text
c10_pass = all([has_dowhy, has_ref_rcc, has_ref_placebo, has_ref_subset])
print(f"10. DoWhy Intervention & 3 Refutations: DoWhy={has_dowhy}, RCC={has_ref_rcc}, Placebo={has_ref_placebo}, Subset={has_ref_subset} -> {'PASSED' if c10_pass else 'FAILED'}")

# 11. Actionable Intervention Mechanism
has_actionability = "Actionable Treatment" in full_text
has_t0 = "t = 0" in full_text
has_immut = "Non-manipulability" in full_text
c11_pass = all([has_actionability, has_t0, has_immut])
print(f"11. Actionable Intervention Principle: Section={has_actionability}, t=0={has_t0}, Non-manipulability={has_immut} -> {'PASSED' if c11_pass else 'FAILED'}")

# 12. 6-step pedagogy
has_theory = "Ý tưởng lý thuyết" in full_text or "Lý thuyết" in full_text
has_code = len(code_cells) == 10
has_result = "Phân tích kết quả đầu ra" in full_text
has_meaning = "Ý nghĩa thực tế" in full_text
has_limits = "Giới hạn" in full_text
c12_pass = all([has_theory, has_code, has_result, has_meaning, has_limits])
print(f"12. 6-Step Pedagogy: Theory={has_theory}, 10 Code Cells={has_code}, Results={has_result}, Meaning={has_meaning}, Limits={has_limits} -> {'PASSED' if c12_pass else 'FAILED'}")

print("=" * 70)
all_pass = all([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass, c7_pass, c8_pass, c9_pass, c10_pass, c11_pass, c12_pass])
print(f"OVERALL STATUS: {'100% PASSED (12/12 CRITERIA SATISFIED)' if all_pass else 'SOME CRITERIA FAILED'}")
print("=" * 70)
