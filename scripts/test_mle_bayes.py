from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import pandas as pd

df = pd.DataFrame({
    'A': [0, 1, 0, 1, 1],
    'B': [1, 0, 1, 0, 1]
})
model = DiscreteBayesianNetwork([('A', 'B')])

# MLE
mle = MaximumLikelihoodEstimator(model, df)
cpd_b_mle = mle.estimate_cpd('B')
print("MLE CPD:")
print(cpd_b_mle)

# Bayesian
bayes = BayesianEstimator(model, df)
cpd_b_bayes = bayes.estimate_cpd('B', prior_type='BDeu', equivalent_sample_size=10)
print("Bayesian CPD:")
print(cpd_b_bayes)
