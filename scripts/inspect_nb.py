import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('00_MASTER_Causal_PGM_Educational_Intervention.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [1, 2, 6, 7, 8, 11, 12]:
    cell = nb['cells'][i]
    print(f"=== Cell {i} ({cell['cell_type']}) ===")
    print(''.join(cell['source'])[:300])
    print('-----------------------------------------')
