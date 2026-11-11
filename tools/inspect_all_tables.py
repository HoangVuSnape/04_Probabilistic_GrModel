import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== TABLE_ALTERNATIVE_OUTCOMES.CSV ===")
df_ao = pd.read_csv('artifacts/tables/Table_Alternative_Outcomes.csv')
for idx, r in df_ao.iterrows():
    print(dict(r))

print("\n=== TABLE_SENSITIVITY_CINELLI.CSV ===")
df_sens = pd.read_csv('artifacts/tables/Table_Sensitivity_Cinelli.csv')
print(df_sens)

print("\n=== TABLE_A_PREDICTION.CSV ===")
df_a = pd.read_csv('artifacts/tables/Table_A_Prediction.csv')
print(df_a)

print("\n=== TABLE_B_INTERVENTION.CSV ===")
df_b = pd.read_csv('artifacts/tables/Table_B_Intervention.csv')
print(df_b)

print("\n=== TABLE_C_ROBUSTNESS_COST.CSV ===")
df_c = pd.read_csv('artifacts/tables/Table_C_Robustness_Cost.csv')
print(df_c)

print("\n=== TABLE_FACTORIAL_H3.CSV ===")
df_h3 = pd.read_csv('artifacts/tables/Table_Factorial_H3.csv')
print(df_h3)
