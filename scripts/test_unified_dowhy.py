import sys
import pandas as pd
import dowhy
from dowhy import CausalModel

sys.stdout.reconfigure(encoding='utf-8')

# Read UCI data
df = pd.read_csv('data/uci_697/data.csv', sep=',')
df.columns = [c.strip().replace(' ', '_').replace("'", "").replace('(', '').replace(')', '').replace('/', '_') for c in df.columns]
df_bin = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
df_bin['Target_Graduate'] = (df_bin['Target'] == 'Graduate').astype(int)

# 5 unified confounders + 1 treatment + 1 mediator + 1 outcome
causal_graph = """digraph {
    Mothers_qualification -> Scholarship_holder;
    Admission_grade -> Scholarship_holder;
    Debtor -> Scholarship_holder;
    Tuition_fees_up_to_date -> Scholarship_holder;
    Age_at_enrollment -> Scholarship_holder;
    
    Mothers_qualification -> Target_Graduate;
    Admission_grade -> Target_Graduate;
    Debtor -> Target_Graduate;
    Tuition_fees_up_to_date -> Target_Graduate;
    Age_at_enrollment -> Target_Graduate;
    
    Scholarship_holder -> Curricular_units_1st_sem_approved;
    Curricular_units_1st_sem_approved -> Target_Graduate;
    Scholarship_holder -> Target_Graduate;
}"""

model = CausalModel(
    data=df_bin,
    treatment='Scholarship_holder',
    outcome='Target_Graduate',
    graph=causal_graph
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
backdoor_vars = identified_estimand.get_backdoor_variables()
print("Identified Backdoor Set (5 variables):", backdoor_vars)
assert len(backdoor_vars) == 5, f"Expected 5 backdoor variables, got {len(backdoor_vars)}"

estimate_lr = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')
print("DoWhy Linear Regression ATE:", round(estimate_lr.value, 4))

estimate_psw = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_weighting')
print("DoWhy Propensity Score Weighting ATE:", round(estimate_psw.value, 4))

ref_common = model.refute_estimate(identified_estimand, estimate_lr, method_name='random_common_cause', num_simulations=5)
print("Refutation 1 (Random Common Cause):", round(ref_common.new_effect, 4))

ref_placebo = model.refute_estimate(identified_estimand, estimate_lr, method_name='placebo_treatment_refuter', num_simulations=5)
print("Refutation 2 (Placebo Treatment):", round(ref_placebo.new_effect, 4))

ref_subset = model.refute_estimate(identified_estimand, estimate_lr, method_name='data_subset_refuter', num_simulations=5, subset_fraction=0.8)
print("Refutation 3 (Data Subset 80%):", round(ref_subset.new_effect, 4))
