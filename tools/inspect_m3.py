import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb = json.loads(Path('00_MASTER_Causal_PGM_Educational_Intervention.ipynb').read_text(encoding='utf-8'))
cells = nb['cells']

print('=== CELL 05 SOURCE ===')
print(''.join(cells[5].get('source', [])))

print('=== CELL 05 OUTPUT ===')
for o in cells[5].get('outputs', []):
    print(''.join(o.get('text', [])))

print('=== CELL 06 SOURCE ===')
print(''.join(cells[6].get('source', [])))

print('=== CELL 30 TABLE 11.1 ===')
c30_text = ''.join(cells[30].get('source', []))
for line in c30_text.splitlines():
    if 'Bảng 11.1' in line or '11.1' in line or 'Naive' in line or 'ATE' in line or 'chênh lệch' in line or '86.17' in line or '34.54' in line:
        print(line)

print('\n=== CELL 22 OUTPUT ===')
c22 = cells[22]
print('Outputs count:', len(c22.get('outputs', [])))
for idx, o in enumerate(c22.get('outputs', [])):
    txt = ''.join(o.get('text', []))
    data_txt = ''.join(o.get('data', {}).get('text/plain', []))
    print(f'Output {idx}: text len={len(txt)}, data len={len(data_txt)}')
    print('text preview:', txt[:200])
    print('data preview:', data_txt[:200])
