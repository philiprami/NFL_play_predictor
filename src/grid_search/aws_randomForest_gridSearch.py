from sklearn import grid_search
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from spark_sklearn import GridSearchCV

df = pd.read_csv('../data/master_RushPassOnly.csv')
y = df.pop('IsPass').values
X = df.values

param_grid = {"max_depth": [3, 5, 10, None],
              "max_features": [None, 'auto', 'log2'],
              "n_estimators": [100],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}

rf = RandomForestClassifier(verbose=2, n_jobs=-1)
gs = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
gs.fit(X, y)
best_parameters = gs.best_params_
print best_parameters