import re
from pathlib import Path

base_dir = Path(r"e:\01_Research_Study_Master\04_Probabilistic_GrModel\src_latex")

def wrap_table_in_file(rel_path, table_label):
    p = base_dir / rel_path
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the table with table_label
    # Wrap \begin{tabular}... \end{tabular} inside \resizebox{\textwidth}{!}{...}
    tab_match = re.search(r'(\\begin\{tabular\}.*?\\end\{tabular\})', text, re.DOTALL)
    if tab_match:
        old_tab = tab_match.group(1)
        if r'\resizebox{\textwidth}{!}' not in old_tab:
            new_tab = r'\resizebox{\textwidth}{!}{%' + '\n' + old_tab + '%\n}'
            text = text.replace(old_tab, new_tab)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Wrapped table in {rel_path}")
        else:
            print(f"Already wrapped in {rel_path}")

if __name__ == '__main__':
    wrap_table_in_file("2_chapters/4_Experiments/4_2_HeldOut_Predictive_Performance.tex", "tab:table_a_prediction")
    wrap_table_in_file("2_chapters/4_Experiments/4_3_Causal_Interventions_and_ATE.tex", "tab:table_b_intervention")
    wrap_table_in_file("2_chapters/4_Experiments/4_4_DoWhy_Refutations_and_Computational_Efficiency.tex", "tab:table_c_robustness")
    wrap_table_in_file("2_chapters/5_Advanced_Analysis/5_1_Alternative_Academic_Outcomes.tex", "tab:alternative_outcomes")
