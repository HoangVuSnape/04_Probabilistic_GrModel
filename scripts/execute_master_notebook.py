import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def execute_notebook():
    project_dir = Path(__file__).resolve().parent.parent
    nb_path = project_dir / 'notebook' / '00_MASTER_Causal_PGM_Educational_Intervention.ipynb'
    
    print(f"[*] Loading notebook: {nb_path}")
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    print("[*] Executing notebook with NotebookClient...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name='python3',
        resources={'metadata': {'path': str(project_dir)}}
    )
    
    client.execute()
    print("[*] Execution complete! Saving notebook with outputs...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"[SUCCESS] Executed notebook saved to: {nb_path}")

if __name__ == '__main__':
    execute_notebook()
