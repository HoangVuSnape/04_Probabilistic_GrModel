import json
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== INSPECTING MASTER NOTEBOOK CELLS ===")
with open("00_MASTER_Causal_PGM_Educational_Intervention.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("\n--- CELL 05 (Code) ---")
print("".join(nb['cells'][5]['source']))
print("Outputs:")
for out in nb['cells'][5].get('outputs', []):
    print(out.get('text', ''))

print("\n--- CELL 06 (Markdown) ---")
print("".join(nb['cells'][6]['source']))

print("\n--- CELL 22 (Code) ---")
print("".join(nb['cells'][22]['source']))

print("\n--- CELL 30 (Markdown) ---")
print("".join(nb['cells'][30]['source']))

print("\n=== COLUMNS OF Table_Multi_Treatment.csv ===")
df_mt = pd.read_csv("artifacts/tables/Table_Multi_Treatment.csv")
print("Columns:", df_mt.columns.tolist())
print(df_mt.head(3))

print("\n=== COLUMNS OF Table_Alternative_Outcomes.csv ===")
df_ao = pd.read_csv("artifacts/tables/Table_Alternative_Outcomes.csv")
print("Columns:", df_ao.columns.tolist())
print(df_ao.head(5))

print("\n=== COLUMNS OF Table_Sensitivity_Cinelli.csv ===")
df_sens = pd.read_csv("artifacts/tables/Table_Sensitivity_Cinelli.csv")
print("Columns:", df_sens.columns.tolist())
print(df_sens)
