from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
import pandas as pd

df = pd.DataFrame({
    'A': [0, 1, 0, 1, 1],
    'B': [1, 0, 1, 0, 1]
})
model = DiscreteBayesianNetwork([('A', 'B')])
estimator = BayesianEstimator(model, df)
cpd_b = estimator.estimate_cpd('B', prior_type='BDeu', equivalent_sample_size=10)
print("Estimated CPD with BDeu:")
print(cpd_b)
