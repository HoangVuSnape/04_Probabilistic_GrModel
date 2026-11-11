import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebook/causal_modeling_uci697.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(12, len(nb['cells'])):
    c = nb['cells'][idx]
    print(f"\n=== CELL {idx} ({c['cell_type']}) ===")
    src = "".join(c.get('source', []))
    print(src[:250].replace("\n", " "))
