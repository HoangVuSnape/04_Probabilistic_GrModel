import json
import hashlib
from pathlib import Path

p_root = Path('00_MASTER_Causal_PGM_Educational_Intervention.ipynb')
p_sub = Path('notebook/00_MASTER_Causal_PGM_Educational_Intervention.ipynb')

assert p_root.exists(), "Root notebook does not exist!"
assert p_sub.exists(), "Sub notebook does not exist!"

b_root = p_root.read_bytes()
b_sub = p_sub.read_bytes()

hash_root = hashlib.sha256(b_root).hexdigest()
hash_sub = hashlib.sha256(b_sub).hexdigest()

print(f"File Size Root: {len(b_root):,} bytes | SHA256: {hash_root[:16]}")
print(f"File Size Sub : {len(b_sub):,} bytes | SHA256: {hash_sub[:16]}")
print(f"Files are identical: {hash_root == hash_sub}")

nb = json.loads(b_root.decode('utf-8'))
cells = nb['cells']
print(f"Total cells: {len(cells)}")

code_cells = [c for c in cells if c['cell_type'] == 'code']
md_cells = [c for c in cells if c['cell_type'] == 'markdown']

print(f"Markdown cells: {len(md_cells)}")
print(f"Code cells    : {len(code_cells)}")

cells_with_output = [c for c in code_cells if c.get('outputs') and len(c['outputs']) > 0]
print(f"Code cells with outputs: {len(cells_with_output)} / {len(code_cells)}")

for i, c in enumerate(code_cells, 1):
    output_types = [o.get('output_type') for o in c.get('outputs', [])]
    print(f"Code Cell {i:02d}: {len(c.get('outputs', []))} outputs ({', '.join(output_types)})")
