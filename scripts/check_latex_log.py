import re
import sys
from pathlib import Path

def analyze_log(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 1. Page count
    page_match = re.search(r'Output written on [^\n]+ \((\d+) pages', content)
    pages = int(page_match.group(1)) if page_match else 0

    # 2. Overfull hboxes
    overfull_matches = re.findall(r'Overfull \\hbox \(([0-9.]+)pt too wide\) (?:in paragraph at lines (\d+)--(\d+)|detected at line (\d+))', content)
    overfulls = []
    for m in overfull_matches:
        width = float(m[0])
        lines = m[1] or m[3] or "unknown"
        if width > 1.0: # ignore negligible micro-overfulls <= 1pt
            overfulls.append((width, lines))

    # 3. Float too large
    float_too_large = re.findall(r'LaTeX Warning: Float too large for page by ([0-9.]+)pt on input line (\d+)', content)

    # 4. Undefined references / citations
    undefined_refs = re.findall(r'LaTeX Warning: Reference `([^`]+)\' on page [^\n]+ undefined', content)
    undefined_cites = re.findall(r'LaTeX Warning: Citation `([^`]+)\' on page [^\n]+ undefined', content)

    print("=" * 60)
    print("           LATEX COMPILATION AUDIT REPORT           ")
    print("=" * 60)
    print(f"[*] Total Page Count: {pages} pages (Target: ~40 pages, 38-44)")
    print(f"[*] Overfull \\hbox count (>1pt): {len(overfulls)}")
    if overfulls:
        for w, l in overfulls[:10]:
            print(f"    - Overfull by {w:.2f}pt at line {l}")
        if len(overfulls) > 10:
            print(f"    ... and {len(overfulls) - 10} more.")

    print(f"[*] Float too large warnings: {len(float_too_large)}")
    for excess, line in float_too_large:
        print(f"    - Float exceeds page by {excess}pt on input line {line}")

    print(f"[*] Undefined References: {len(undefined_refs)} {undefined_refs[:5]}")
    print(f"[*] Undefined Citations:  {len(undefined_cites)} {undefined_cites[:5]}")
    print("=" * 60)

    # Check criteria
    is_page_ok = 36 <= pages <= 46
    is_overfull_ok = len(overfulls) == 0
    is_float_ok = len(float_too_large) == 0
    is_refs_ok = len(undefined_refs) == 0 and len(undefined_cites) == 0

    status = is_page_ok and is_overfull_ok and is_float_ok and is_refs_ok
    print(f"[*] OVERALL AUDIT STATUS: {'PASSED' if status else 'NEEDS ITERATION'}")
    return status, pages, len(overfulls), len(float_too_large)

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    log_file = project_dir / 'src_latex' / 'main.log'
    if not log_file.exists():
        print("[!] main.log not found")
        sys.exit(1)
    analyze_log(log_file)
