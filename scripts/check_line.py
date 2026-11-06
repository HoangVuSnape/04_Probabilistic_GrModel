import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('tools/build_master_notebook_v3.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(25, 95):
    print(f'{i+1}: {repr(lines[i])}')
