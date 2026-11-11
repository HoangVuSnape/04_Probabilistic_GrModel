# -*- coding: utf-8 -*-
"""Dựng SO_TAY_BAO_VE.html: 5 tab (kịch bản, bảng & hình, giải thích code, QA, glossary) + tìm kiếm."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fn_annotations import GROUPS, A  # noqa: E402

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NB = ROOT / 'notebook' / '00_MASTER_Causal_PGM_Educational_Intervention.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def read(p):
    return (ROOT / p).read_text(encoding='utf-8')


# ---------------- Tab 2: bảng & hình + giải thích ----------------
DOC_RE = re.compile(r'^#{3,4} .*(Phân tích kết quả|Đọc |Đọc$|Giới hạn của|1\.6c|8\.E2)', re.M)


def outputs_of(i):
    res = []
    for o in cells[i].get('outputs', []):
        d = o.get('data', {})
        if 'image/png' in d:
            res.append({'t': 'img', 'v': d['image/png'].replace('\n', '')})
        elif 'text/html' in d:
            res.append({'t': 'html', 'v': ''.join(d['text/html'])})
        elif 'text/plain' in d and o.get('output_type') == 'execute_result':
            res.append({'t': 'text', 'v': ''.join(d['text/plain'])})
        elif o.get('output_type') == 'stream':
            txt = ''.join(o.get('text', ''))
            if txt.strip():
                res.append({'t': 'text', 'v': txt})
    return res


MANUAL = [
    ([4, 6], [7], None),
    ([9, 11, 12, 13, 14], [15], None),
    ([17, 18], [19], None),
    ([21, 22], [23], None),
    ([25], [26], None),
    ([30, 31], [32, 33], None),
    ([35], [36], None),
    ([42, 43], [], '8.A. Dữ liệu 7 biến và 5 split dùng chung'),
    ([45, 47, 49], [], '8.B–8.D. Kiểm thử nhanh B0, H1, H2 trên seed 42 (chọn hàm điểm, epoch tốt nhất, chọn lambda và tau)'),
    ([52], [], '8.E. Vòng lặp 5 run ghép cặp — ATE của B0, H1, H2 theo từng seed'),
    ([53], [54], None),
    ([58, 64], [65], None),
    ([59], [60], None),
    ([61, 62], [63], None),
    ([69, 71, 73, 75, 77, 79, 81, 83], [84], None),
]
figs = []
for cs, ex, title in MANUAL:
    blocks = [{'cell': i, 'outs': outputs_of(i)} for i in cs if outputs_of(i)]
    explain = [re.sub(r"<a id='[^']+'></a>\n?", '', ''.join(cells[j]['source'])) for j in ex]
    g = {'blocks': blocks, 'explain': explain, 'cells': [b['cell'] for b in blocks]}
    h = re.search(r'^#{3,4} (.*)$', explain[0], re.M) if explain else None
    g['title'] = title or (h.group(1).strip() if h else f'Output cell {cs[0]}')
    figs.append(g)

# ---------------- Tab 3: giải thích code ----------------
code_groups = []
gdir = ROOT / 'report' / 'midterm' / 'giai_thich_function'
SEP = '# ' + '=' * 110
for so, ten, tieu_de, mo_ta, muc in GROUPS:
    if so == 13:
        continue  # nhóm mục 10 cũ (phân bổ chính sách) đã bỏ khỏi notebook
    text = (gdir / f'Group{so:02d}_{ten}.py').read_text(encoding='utf-8')
    parts = text.split(SEP + '\n')
    # parts: [header, ann1, code1, ann2, code2, ...] (mỗi annotation mở và đóng bằng SEP)
    items = []
    for k in range(1, len(parts), 2):
        ann = parts[k]
        code = parts[k + 1].rstrip('\n') if k + 1 < len(parts) else ''
        first = ann.splitlines()[0][2:]
        name = first.split('  (')[0].strip()
        loai = first.split('  (')[1].rstrip(')') if '  (' in first else ''
        vitri = ann.splitlines()[1][2:].replace('Vị trí: ', '')
        cell = int(re.search(r'Cell (\d+)', vitri).group(1))
        key = f'__code_{cell}' if name.startswith('doan_ma') else name
        a = A.get(key, {})
        md = []
        if 'muc' in a: md.append(f"**Mục đích.** {a['muc']}")
        if 'vao' in a: md.append(f"**Đầu vào.** {a['vao']}")
        if 'ra' in a: md.append(f"**Đầu ra.** {a['ra']}")
        if 'cach' in a:
            md.append('**Cách làm:**\n\n' + '\n'.join(f'{n}. {s}' for n, s in enumerate(a['cach'], 1)))
        if 'ly_thuyet' in a: md.append(f"**Lý thuyết.** {a['ly_thuyet']}")
        if 'luu_y' in a: md.append(f"**Lưu ý.** {a['luu_y']}")
        for q, ans in a.get('hoi', []):
            md.append(f"> ❓ **{q}**  \n> ➜ {ans}")
        items.append({'name': name, 'kind': loai, 'where': vitri, 'md': '\n\n'.join(md), 'code': code})
    code_groups.append({'title': f'Nhóm {so:02d} — {tieu_de}', 'file': f'Group{so:02d}_{ten}.py', 'desc': mo_ta, 'items': items})

DATA = {
    'script': read('report/midterm/KICH_BAN_CHON_LOC.md'),
    'script_full': read('report/midterm/KICH_BAN_TRINH_BAY_NOTEBOOK.md'),
    'figs': figs,
    'code': code_groups,
    'qa': read('report/midterm/NOTEBOOK_DEFENSE_QA_SCRIPT.md'),
    'glossary': read('glossary_de_hieu.md'),
    'formulas': read('report/midterm/CONG_THUC_DE_HIEU.md'),
}
payload = json.dumps(DATA, ensure_ascii=False).replace('</', '<\\/')
tpl = (Path(__file__).parent / 'hub_template.html').read_text(encoding='utf-8')
katex_css = (Path(__file__).parent / 'katex_nofont.css').read_text(encoding='utf-8')
html = tpl.replace('/*__KATEX__*/', katex_css).replace('/*__DATA__*/null', payload)
if len(sys.argv) > 3 and sys.argv[3] == '--artifact':
    # Bản đăng artifact: nền tảng tự bọc doctype/html/head/body
    for wrapper in ['<!doctype html>\n', '<html lang="vi">\n', '<head>\n', '<meta charset="utf-8">\n',
                    '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
                    '</head>\n', '<body>\n', '</body>\n', '</html>\n']:
        assert html.count(wrapper) == 1, wrapper
        html = html.replace(wrapper, '')
    assert '<head>' not in html and '<body>' not in html and '</html>' not in html
OUT.write_text(html, encoding='utf-8')
print('wrote', OUT, f'{OUT.stat().st_size / 1e6:.2f} MB', '| fig groups:', len(figs),
      '| code items:', sum(len(g['items']) for g in code_groups))
for g in figs:
    print('  ', g['cells'], '->', g['title'][:70])
