from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, MLE
import pandas as pd

df = pd.DataFrame({
    'A': [0, 1, 0, 1, 1],
    'B': [1, 0, 1, 0, 1]
})
model = DiscreteBayesianNetwork([('A', 'B')])
# Try fit default
model.fit(df)
print("Default fit successful! CPD B:")
print(model.get_cpds('B'))

# Try fit with BayesianEstimator instance or class
estimator = BayesianEstimator(model, df)
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
print("Bayesian fit successful! CPD B:")
print(model.get_cpds('B'))
