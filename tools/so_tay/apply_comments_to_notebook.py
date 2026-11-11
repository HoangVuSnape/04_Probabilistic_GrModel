# -*- coding: utf-8 -*-
"""Chép chú thích # từng dòng (annotated_code.py) vào CHÍNH notebook, không chạy lại.
An toàn: chỉ thay khoảng dòng của hàm / câu lệnh; output giữ nguyên; kiểm tra AST từng cell trước khi ghi."""
import ast, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from annotated_code import A  # noqa: E402

for path in sys.argv[1:]:
    nb = json.load(open(path, encoding='utf-8'))
    doi = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        lines = src.split('\n')
        tree = ast.parse(src)
        thay = []  # (dòng bắt đầu 0-index, dòng kết thúc exclusive, các dòng mới)
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in A:
                st = (n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1
                thay.append((st, n.end_lineno, A[n.name].rstrip('\n').split('\n')))
        key = f'__code_{i}'
        if key in A:  # đoạn mã tuần tự: ghép theo thứ tự từng câu lệnh (bỏ qua import, def, class)
            goc = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
            ann_lines = A[key].split('\n')
            moi = [n for n in ast.parse(A[key]).body]
            assert len(goc) == len(moi), (i, len(goc), len(moi))
            for g, m in zip(goc, moi):
                assert ast.dump(g) == ast.dump(m), f'cell {i}: câu lệnh khác nhau'
                thay.append((g.lineno - 1, g.end_lineno, ann_lines[m.lineno - 1:m.end_lineno]))
        if not thay:
            continue
        for st, en, new in sorted(thay, key=lambda x: -x[0]):  # thay từ dưới lên để không lệch số dòng
            lines[st:en] = new
        moi_src = '\n'.join(lines)
        assert ast.dump(ast.parse(moi_src)) == ast.dump(tree), f'cell {i}: AST thay đổi!'  # bỏ comment thì giống hệt
        c['source'] = moi_src.splitlines(keepends=True)
        doi.append((i, len(thay)))
    json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(path, 'a', encoding='utf-8').write('\n')
    print(path, '-> cell đã thêm chú thích:', doi)
