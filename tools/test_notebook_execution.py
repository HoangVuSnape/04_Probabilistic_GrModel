import json
import sys
import traceback

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

def mock_display(*args, **kwargs):
    for arg in args:
        print(arg)

def run_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    global_ns = {"display": mock_display, "plt": plt}
    
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Executing {len(code_cells)} code cells from {nb_path}...")
    
    for idx, cell in enumerate(code_cells, 1):
        source = "".join(cell.get("source", []))
        first_line = source.strip().split("\n")[0] if source.strip() else ""
        print(f"\n--- [Cell {idx}/{len(code_cells)}] {first_line[:70]} ---")
        try:
            exec(source, global_ns)
            print(f"Cell {idx} SUCCESS")
        except Exception as e:
            print(f"ERROR in Cell {idx}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    return True

if __name__ == "__main__":
    nb_path = sys.argv[1] if len(sys.argv) > 1 else "notebook/causal_modeling_uci697.ipynb"
    success = run_notebook(nb_path)
    if success:
        print(f"\nALL CELLS IN {nb_path} EXECUTED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\nNOTEBOOK EXECUTION FAILED FOR {nb_path}!")
        sys.exit(1)
