import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

print("=== CHECKING FILES IN REPO ===")
for p in ["notebook", "notebooks", "src_latex"]:
    if os.path.exists(p):
        print(f"Directory {p} exists: contents = {os.listdir(p)}")
    else:
        print(f"Directory {p} DOES NOT EXIST")

print("\n=== FINDING ALL .ipynb FILES ===")
for f in glob.glob("**/*.ipynb", recursive=True):
    print(" -", f)

print("\n=== FINDING ALL .tex FILES ===")
for f in glob.glob("src_latex/**/*.tex", recursive=True):
    print(" -", f)
