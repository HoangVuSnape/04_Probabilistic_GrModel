from pathlib import Path

p = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\config\format.tex")
with open(p, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(r"\setcounter{tocdepth}{3}", r"\setcounter{tocdepth}{2}")

with open(p, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated tocdepth to 2 in format.tex")
