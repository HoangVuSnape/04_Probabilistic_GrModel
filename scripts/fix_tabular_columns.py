from pathlib import Path

p_4_5 = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\4_Experiments\4_5_Ablation_Studies.tex")
with open(p_4_5, "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace(r"\begin{tabular}{lp{3.5cm}cccX}", r"\begin{tabular}{lp{3.5cm}cccp{5.5cm}}")
with open(p_4_5, "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed 4_5")

p_3_2 = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\3_Methods_Models\3_2_Factorial_Model_Matrix.tex")
with open(p_3_2, "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace(r"\begin{tabular}{lp{3.5cm}p{3.5cm}X}", r"\begin{tabular}{lp{3.5cm}p{3.5cm}p{5.5cm}}")
with open(p_3_2, "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed 3_2")
