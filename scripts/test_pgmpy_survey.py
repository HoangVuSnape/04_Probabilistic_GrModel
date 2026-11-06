import sys
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, BDeu, MaximumLikelihoodEstimator, BayesianEstimator

sys.stdout.reconfigure(encoding='utf-8')

# Read UCI data
df = pd.read_csv('data/uci_697/data.csv', sep=',')
df.columns = [c.strip().replace(' ', '_').replace("'", "").replace('(', '').replace(')', '').replace('/', '_') for c in df.columns]
df_bin = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
df_bin['Target_Graduate'] = (df_bin['Target'] == 'Graduate').astype(int)

# Select key discrete variables
subset_cols = ['Scholarship_holder', 'Debtor', 'Target_Graduate']
demo_df = df_bin[subset_cols].copy()

# 1. Structure Learning Survey Demo: Hill Climb Search with BIC Score
print("--- 1. Structure Learning Demo (Hill Climb Search with BIC) ---")
hc = HillClimbSearch(demo_df)
scoring = BIC(demo_df)
best_model = hc.estimate(scoring_method=scoring, max_iter=20)
print("Learned edges:", best_model.edges())

# 2. Parameter Learning Survey Demo: MLE vs Bayesian Estimator
print("--- 2. Parameter Learning Demo (MLE vs Bayesian Estimator) ---")
model_mle = DiscreteBayesianNetwork([('Scholarship_holder', 'Target_Graduate'), ('Debtor', 'Target_Graduate')])
model_mle.fit(demo_df, estimator=MaximumLikelihoodEstimator)
cpd_target_mle = model_mle.get_cpds('Target_Graduate')
print("MLE CPD for Target_Graduate:\n", cpd_target_mle)

model_bayes = DiscreteBayesianNetwork([('Scholarship_holder', 'Target_Graduate'), ('Debtor', 'Target_Graduate')])
model_bayes.fit(demo_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
cpd_target_bayes = model_bayes.get_cpds('Target_Graduate')
print("Bayesian (BDeu) CPD for Target_Graduate:\n", cpd_target_bayes)
