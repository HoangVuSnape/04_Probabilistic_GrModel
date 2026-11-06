import json

with open('00_MASTER_Causal_PGM_Educational_Intervention.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

for i, cell in enumerate(code_cells):
    n_out = len(cell.get('outputs', []))
    print(f"Code cell {i+1}: {n_out} outputs recorded.")
