# -*- coding: utf-8 -*-
import ast, json, re, sys, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fn_annotations import GROUPS, A  # noqa: E402
from annotated_code import A as ANN_CODE  # noqa: E402  (bản code có chú thích # từng dòng)


def same_ast(a: str, b: str) -> bool:
    """Bỏ comment thì hai đoạn code phải giống hệt về cấu trúc (AST)."""
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))

NB, OUTDIR = Path(sys.argv[1]), Path(sys.argv[2])
NB_NAME = '00_MASTER_Causal_PGM_Educational_Intervention.ipynb'
OUTDIR.mkdir(parents=True, exist_ok=True)
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']

# Mục (tiêu đề markdown gần nhất phía trước) cho từng cell
section, cur = {}, ''
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown':
        for ln in ''.join(c['source']).splitlines():
            m = re.match(r'^#{2,4} (\d+[\.\w]*\.?) (.*)$', ln)
            if m and not re.match(r'^\d+\.[A-Z]\.\d', m.group(1)):
                cur = (m.group(1).rstrip('.') + ' ' + re.sub(r'\s*\(Slide[^)]*\)', '', m.group(2))).strip()
    section[i] = cur
section[4] = '1.2–1.4 Metadata, nạp và kiểm toán dữ liệu UCI 697'  # cell code nằm ngay sau 1.1b nhưng thuộc 1.2–1.4

# Trích hàm/class nguyên văn
defs = {}
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    lines = src.splitlines()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            start = (n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1
            defs[n.name] = (i, '\n'.join(lines[start:n.end_lineno]))


def code_block(i: int) -> str:
    """Các câu lệnh cấp cao nhất không phải def/class/import của cell i, giữ nguyên thứ tự."""
    src = ''.join(cells[i]['source'])
    lines = src.splitlines()
    parts = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        parts.append('\n'.join(lines[n.lineno - 1:n.end_lineno]))
    return '\n'.join(parts)


def wrap(prefix: str, text: str, width: int = 112) -> list:
    return textwrap.wrap(text, width=width, initial_indent=prefix, subsequent_indent='#' + ' ' * (len(prefix) - 1),
                         break_long_words=False, break_on_hyphens=False)


def ann_block(key: str, title: str, cell: int) -> str:
    a = A.get(key, {})
    out = ['# ' + '=' * 110, f'# {title}', f'# Vị trí: Cell {cell} · mục {section.get(cell, "")}', '# ' + '-' * 110]
    if 'muc' in a:
        out += wrap('# MỤC ĐÍCH : ', a['muc'])
    if 'vao' in a:
        out += wrap('# ĐẦU VÀO  : ', a['vao'])
    if 'ra' in a:
        out += wrap('# ĐẦU RA   : ', a['ra'])
    if 'cach' in a:
        out.append('# CÁCH LÀM :')
        for k, s in enumerate(a['cach'], 1):
            out += wrap(f'#   {k}. ', s)
    if 'ly_thuyet' in a:
        out += wrap('# LÝ THUYẾT: ', a['ly_thuyet'])
    if 'luu_y' in a:
        out += wrap('# LƯU Ý    : ', a['luu_y'])
    for q, ans in a.get('hoi', []):
        out += wrap('# ❓ HỎI   : ', q)
        out += wrap('#   ➜ ĐÁP : ', ans)
    out.append('# ' + '=' * 110)
    return '\n'.join(out)


index_rows = []  # cho file md
missing = []
for so, ten, tieu_de, mo_ta, muc in GROUPS:
    fname = f'Group{so:02d}_{ten}.py'
    body = []
    noi_dung = []
    for kind, *rest in muc:
        if kind == 'fn':
            name = rest[0]
            if name not in defs:
                missing.append(name)
                continue
            cell, src = defs[name]
            if name in ANN_CODE:
                assert same_ast(ANN_CODE[name], src), f'Chú thích làm đổi code của {name}!'
                src = ANN_CODE[name].rstrip()
            loai = 'class' if src.lstrip().startswith(('class', '@')) and 'class ' + name in src.split('\n', 3)[0:3].__str__() else 'def'
            body.append(ann_block(name, f'{name}  ({loai})', cell) + '\n' + src)
            noi_dung.append((name, cell, loai))
            index_rows.append((so, fname, name, cell, loai))
        else:
            cell, ten_khoi = rest
            key = f'__code_{cell}'
            fn_name = f'doan_ma_cell_{cell}_{ten_khoi}'
            goc = code_block(cell)
            if key in ANN_CODE:
                assert same_ast(ANN_CODE[key], goc), f'Chú thích làm đổi code của cell {cell}!'
                goc = ANN_CODE[key]
            code = textwrap.indent(goc, '    ')
            note = ('    # (Trong notebook đây là code tuần tự ở cấp cao nhất của cell; ở file này được bọc trong một hàm\n'
                    '    #  CHỈ ĐỂ ĐỌC, nội dung giữ nguyên văn.)\n')
            body.append(ann_block(key, f'{fn_name}  (đoạn mã tuần tự)', cell) + f'\ndef {fn_name}():\n' + note + code)
            noi_dung.append((fn_name, cell, 'đoạn mã'))
            index_rows.append((so, fname, fn_name, cell, 'đoạn mã'))
    header = [
        '# -*- coding: utf-8 -*-',
        '"""',
        f'NHÓM {so:02d} — {tieu_de}',
        '',
        textwrap.fill(mo_ta, 110),
        '',
        f'Code được TRÍCH NGUYÊN VĂN từ notebook/{NB_NAME}. Với các hàm phức tạp (mục 8–9), mỗi dòng có thêm',
        'chú thích "# ..." ở cuối dòng; đã kiểm tra tự động: bỏ chú thích đi thì code giống hệt notebook (so sánh AST).',
        'Phần chú thích tiếng Việt phía trên mỗi hàm là giải thích thêm để trả lời khi bị hỏi sâu.',
        'File này để ĐỌC, không chạy độc lập: các hàm dựa vào import và biến toàn cục của notebook',
        '(pd, np, torch, GLOBAL_CONFIG, STATE_CARD, run_artifacts, ...).',
        '',
        'Nội dung:',
    ] + [f'  - {n:<42} Cell {c:>2} ({l})' for n, c, l in noi_dung] + ['"""', '']
    text = '\n'.join(header) + '\n\n' + '\n\n\n'.join(body) + '\n'
    ast.parse(text)  # bảo đảm là Python hợp lệ
    (OUTDIR / fname).write_text(text, encoding='utf-8')
    print('wrote', fname, len(noi_dung), 'mục')

# kiểm tra: mọi hàm trong notebook đều có mặt
all_names = set(defs)
used = {r[2] for r in index_rows}
print('missing in notebook:', missing)
print('notebook functions not in any group:', sorted(all_names - used))
print('functions without annotation:', sorted(n for n in used if n not in A and not n.startswith('doan_ma')))

# ----------------------------- file md -----------------------------
md = ['# Giải thích các hàm trong notebook chính', '',
      f'Notebook: `notebook/{NB_NAME}`. Các hàm được chia thành {len(GROUPS)} nhóm theo chức năng; mỗi nhóm là một file '
      '`GroupNN_TenNhom.py` trong thư mục này, chứa **code nguyên văn** của notebook kèm chú thích chi tiết '
      '(mục đích, đầu vào/ra, từng bước, lý thuyết, câu hỏi hay gặp). File `.md` này là bảng tra cứu nhanh.', '',
      '> Code trong các file `.py` không được sửa so với notebook; mọi giải thích nằm trong comment. '
      'Notebook không bị thay đổi.', '',
      '## Bảng nhóm', '', '| Nhóm | File | Nội dung | Mục notebook |', '| :--- | :--- | :--- | :--- |']
for so, ten, tieu_de, mo_ta, muc in GROUPS:
    fname = f'Group{so:02d}_{ten}.py'
    cs = sorted({r[3] for r in index_rows if r[0] == so})
    md.append(f'| {so:02d} | [{fname}]({fname}) | {tieu_de} | Cell {cs[0]}–{cs[-1]} |')
md += ['', '## Tra cứu theo hàm', '']
for so, ten, tieu_de, mo_ta, muc in GROUPS:
    fname = f'Group{so:02d}_{ten}.py'
    md += [f'### Nhóm {so:02d} — {tieu_de}', f'File: [{fname}]({fname}). {mo_ta}', '']
    for r in [r for r in index_rows if r[0] == so]:
        name, cell = r[2], r[3]
        key = f'__code_{cell}' if name.startswith('doan_ma') else name
        a = A.get(key, {})
        md.append(f'- **`{name}`** · Cell {cell} · {r[4]} — {a.get("muc", "")}')
        for q, ans in a.get('hoi', []):
            md.append(f'  - *Hỏi:* {q} → *Đáp:* {ans}')
    md.append('')
md += ['## Luồng gọi chính của mục 8 (B0 / H1 / H2)', '', '```text',
       'load_and_preprocess_uci697 -> get_or_create_splits            (Group05)',
       'cho mỗi seed 42..46 (doan_ma_cell_52_vong_lap_5_run_ghep_cap, Group09):',
       '  B0: learn_b0_dag -> fit_bn_on_dag                          (Group06)',
       '  H1: train_h1_mlp -> build_h1_bayesian_network              (Group07, dùng DAG của B0)',
       '  H2: train_notears_mlp -> select_h2_graph -> fit_bn_on_dag  (Group08)',
       '  mỗi mô hình: record_model_run -> query_model_effects       (Group09, Group06)',
       '               -> evaluate_prediction_metrics / compute_bootstrap_ate_ci / identify_with_dowhy / overlap_report',
       'xuất Table A, B (Group09) -> 8.E2 overfit (Group10) -> refuter, Table C (Group11)',
       '```', '']
(OUTDIR / 'GIAI_THICH_FUNCTION.md').write_text('\n'.join(md), encoding='utf-8')
print('wrote GIAI_THICH_FUNCTION.md')
