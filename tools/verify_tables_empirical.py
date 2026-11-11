import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('artifacts/tables/Table_Multi_Treatment.csv')

print("--- POOLED 5 RUNS ---")
pooled = df[df['seed'] == 'Pooled 5 Runs']
for idx, r in pooled.iterrows():
    print(f"{r['treatment_id']:3s} | {r['estimator']:35s} | Risk0={r['risk_do_0']*100:5.2f}% | Risk1={r['risk_do_1']*100:5.2f}% | ATE={r['ate']*100:+6.2f}% | CI=[{r['ci_95_low']*100:5.2f}%, {r['ci_95_high']*100:5.2f}%] | p={r['p_value']:.2e}")

print("\n--- FULL COHORT (N=3,630) ---")
fc = df[df['cohort'] == 'Full Cohort (N=3,630)']
for idx, r in fc.iterrows():
    print(f"{r['treatment_id']:3s} | {r['estimator']:35s} | Risk0={r['risk_do_0']*100:5.2f}% | Risk1={r['risk_do_1']*100:5.2f}% | ATE={r['ate']*100:+6.2f}% | CI=[{r['ci_95_low']*100:5.2f}%, {r['ci_95_high']*100:5.2f}%] | p={r['p_value']:.2e}")
