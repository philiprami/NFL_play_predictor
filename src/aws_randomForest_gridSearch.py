from sklearn import grid_search
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from spark_sklearn import GridSearchCV

df = pd.read_csv('../data/master_RushPassOnly.csv')
y = df.pop('IsPass').values
X = df.values

param_grid = {"max_depth": [5, 10],
              "max_features": [None, 'auto'],
              "n_estimators": [1000, 5000]}

gs = grid_search.GridSearchCV(RandomForestClassifier(verbose=2, n_jobs=-1), param_grid=param_grid, n_jobs=-1, verbose=2)
gs.fit(X, y)
best_parameters = gs.best_params_
print best_parameters