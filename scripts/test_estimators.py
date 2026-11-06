import pgmpy.estimators as pe

print([x for x in dir(pe) if 'MLE' in x or 'Bayes' in x])
