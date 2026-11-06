from pathlib import Path

p = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\5_Advanced_Analysis\5_4_Cinelli_Hazlett_Sensitivity_Contour.tex")
with open(p, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(r"\begin{tabular}{lp{4.2cm}cX}", r"\begin{tabular}{lp{4.2cm}cp{6.5cm}}")

with open(p, "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed table 5.3 columns in 5_4")
