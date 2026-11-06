import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Read UCI data
df = pd.read_csv('data/uci_697/data.csv', sep=',')
df.columns = [c.strip().replace(' ', '_').replace("'", "").replace('(', '').replace(')', '').replace('/', '_') for c in df.columns]
df_bin = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
df_bin['Target_Graduate'] = (df_bin['Target'] == 'Graduate').astype(int)

# Check 5 unified confounders
confounders = ['Admission_grade', 'Age_at_enrollment', 'Debtor', 'Tuition_fees_up_to_date', 'Mothers_qualification']
for col in confounders:
    assert col in df_bin.columns, f"Missing {col}"

print("All 5 confounders present in df_bin!")
print("df_bin shape:", df_bin.shape)
print("Target distribution:\n", df_bin['Target_Graduate'].value_counts())
print("Scholarship distribution:\n", df_bin['Scholarship_holder'].value_counts())
