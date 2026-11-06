import re
from pathlib import Path

def update_format_tex():
    path = Path(r'e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex\config\format.tex')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fontsize 12
    content = content.replace(r'\usepackage[fontsize=13]{scrextend}', r'\usepackage[fontsize=12]{scrextend}')
    # Geometry: 3cm left, 2cm right, 2.5cm top, 2.5cm bottom
    content = re.sub(r'\\usepackage\[left=3\.5cm,\s*right=2cm,\s*top=3\.5cm,\s*bottom=3cm\]\{geometry\}',
                     r'\\usepackage[left=3.0cm, right=2.0cm, top=2.5cm, bottom=2.5cm]{geometry}', content)
    # Column types
    if r'\newcolumntype{Y}' not in content:
        content = content.replace(r'\usepackage{array,tabularx,booktabs}',
                                  r'\usepackage{array,tabularx,booktabs}' + '\n' +
                                  r'\newcolumntype{Y}{>{\centering\arraybackslash}X}' + '\n' +
                                  r'\newcolumntype{L}{>{\raggedright\arraybackslash}X}')

    # Titlespacing
    content = re.sub(r'\\titlespacing\*\{\\chapter\}\{0pt\}\{40pt\}\{20pt\}',
                     r'\\titlespacing*{\\chapter}{0pt}{16pt}{10pt}', content)
    content = re.sub(r'\\titlespacing\*\{\\section\}\{0pt\}\{12pt\}\{6pt\}',
                     r'\\titlespacing*{\\section}{0pt}{10pt}{4pt}', content)
    content = re.sub(r'\\titlespacing\*\{\\subsection\}\{0pt\}\{10pt\}\{4pt\}',
                     r'\\titlespacing*{\\subsection}{0pt}{8pt}{3pt}', content)
    content = re.sub(r'\\titlespacing\*\{\\subsubsection\}\{0pt\}\{8pt\}\{4pt\}',
                     r'\\titlespacing*{\\subsubsection}{0pt}{6pt}{2pt}', content)

    # Baselinestretch and indent
    content = re.sub(r'\\renewcommand\{\\baselinestretch\}\{1\.5\}', r'\\renewcommand{\\baselinestretch}{1.20}', content)
    content = re.sub(r'\\setlength\{\\parindent\}\{1\.27cm\}', r'\\setlength{\\parindent}{0.75cm}', content)

    # Float spacing
    if r'\setlength{\textfloatsep}' not in content:
        float_spacing = ('\n\\setlength{\\textfloatsep}{10pt plus 2pt minus 2pt}\n'
                         '\\setlength{\\floatsep}{8pt plus 2pt minus 2pt}\n'
                         '\\setlength{\\intextsep}{8pt plus 2pt minus 2pt}\n')
        content = content.replace(r'\usepackage[labelfont=bf]{caption}', float_spacing + r'\usepackage[labelfont=bf]{caption}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[SUCCESS] Updated config/format.tex")

if __name__ == '__main__':
    update_format_tex()
