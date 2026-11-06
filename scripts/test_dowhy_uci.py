import sys
import pandas as pd
import dowhy
from dowhy import CausalModel

sys.stdout.reconfigure(encoding='utf-8')

# Read UCI data
df = pd.read_csv('data/uci_697/data.csv', sep=',')
print('UCI shape:', df.shape)

# Clean column names
df.columns = [c.strip().replace(' ', '_').replace("'", "").replace('(', '').replace(')', '').replace('/', '_') for c in df.columns]
df_bin = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
df_bin['Target_Graduate'] = (df_bin['Target'] == 'Graduate').astype(int)

# 7-node domain DAG in DOT format
causal_graph = """digraph {
    Age_at_enrollment -> Scholarship_holder;
    Debtor -> Scholarship_holder;
    Previous_qualification_grade -> Scholarship_holder;
    Scholarship_holder -> Curricular_units_1st_sem_approved;
    Scholarship_holder -> Target_Graduate;
    Curricular_units_1st_sem_approved -> Target_Graduate;
    Age_at_enrollment -> Target_Graduate;
    Debtor -> Target_Graduate;
    Previous_qualification_grade -> Target_Graduate;
}"""

model = CausalModel(
    data=df_bin,
    treatment='Scholarship_holder',
    outcome='Target_Graduate',
    graph=causal_graph
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("Identified Backdoor Set:", identified_estimand.get_backdoor_variables())

estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')
print("DoWhy Linear Regression ATE:", round(estimate.value, 4))

estimate_psw = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_weighting')
print("DoWhy Propensity Score Weighting ATE:", round(estimate_psw.value, 4))

ref_common = model.refute_estimate(identified_estimand, estimate, method_name='random_common_cause', num_simulations=5)
print("Refutation 1 (Random Common Cause):", round(ref_common.new_effect, 4))

ref_placebo = model.refute_estimate(identified_estimand, estimate, method_name='placebo_treatment_refuter', num_simulations=5)
print("Refutation 2 (Placebo Treatment):", round(ref_placebo.new_effect, 4))

ref_subset = model.refute_estimate(identified_estimand, estimate, method_name='data_subset_refuter', num_simulations=5, subset_fraction=0.8)
print("Refutation 3 (Data Subset 80%):", round(ref_subset.new_effect, 4))
