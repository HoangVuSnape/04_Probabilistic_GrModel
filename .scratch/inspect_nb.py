import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("00_MASTER_Causal_PGM_Educational_Intervention.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        for line in source.split("\n"):
            if line.startswith("#"):
                print(f"[{i:02d}] {line}")
