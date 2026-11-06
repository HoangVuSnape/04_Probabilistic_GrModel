from pathlib import Path

# 1. Update 07_ListOfFigures.tex
p_lof = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\1_frontmatter\07_ListOfFigures.tex")
with open(p_lof, "w", encoding="utf-8") as f:
    f.write(r"""\renewcommand{\listfigurename}{LIST OF FIGURES}
\listoffigures\label{listfigures}
\addcontentsline{toc}{chapter}{\listfigurename}

\vspace{1.2cm}
\renewcommand{\listtablename}{LIST OF TABLES}
\listoftables\label{listtables}
\addcontentsline{toc}{chapter}{\listtablename}
\newpage
""")
print("Updated 07_ListOfFigures.tex")

# 2. Update main.tex to comment out 08_ListOfTables
p_main = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\main.tex")
with open(p_main, "r", encoding="utf-8") as f:
    t_main = f.read()
t_main = t_main.replace(r"\input{1_frontmatter/08_ListOfTables}", r"% \input{1_frontmatter/08_ListOfTables} % Combined with List of Figures")
with open(p_main, "w", encoding="utf-8") as f:
    f.write(t_main)
print("Updated main.tex")

# 3. Update 1_3 paragraph to bold
p_1_3 = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\2_chapters\1_Introduction\1_3_Research_questions_and_hypotheses.tex")
with open(p_1_3, "r", encoding="utf-8") as f:
    t_1_3 = f.read()
t_1_3 = t_1_3.replace(r"\paragraph{Overarching Question (RQ0):}", r"\noindent\textbf{Overarching Question (RQ0):}")
with open(p_1_3, "w", encoding="utf-8") as f:
    f.write(t_1_3)
print("Updated 1_3")
