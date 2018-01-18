import numpy as np
import pandas as pd
import IPython
import pymongo
from pymongo import MongoClient

from itertools import groupby
from time import time
from IPython import parallel

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.cross_validation import KFold

def get_mongodb():
    """Get MongoDB database. """
    db_cilent = MongoClient()
    db = db_cilent.foobar
    return db

def summarize_results():
    """Get top grid cells from mongodb.
    Averages the staged scores for each grid cell and picks the best
    setting for ``n_estimators``.
    """
    db = get_mongodb()

    results = db.grid_search.find().sort('grid_cell', pymongo.ASCENDING)
    results = sorted(results, key=lambda x: x['grid_cell'])
    print("got %d results" % len(results))
    out = []

    for grid_cell, group in groupby(results, lambda x: x['grid_cell']):
        group = list(group)
        n_folds = len(group)
        A = np.row_stack([g['scores'] for g in group])
        scores = A.mean(axis=0)
        best_iter = np.argmin(scores)
        best_score = scores[best_iter]
        params = group[0]['params']
        params['n_estimators'] = best_iter + 1
        out.append({'best_score': best_score, 'grid_cell': grid_cell,
                    'params': params, 'n_folds': n_folds})

    out = sorted(out, key=lambda x: x['best_score'])
    print(out[:10])
    return out

def _parallel_grid_search(args):
    """Evaluate parameter grid cell.
    Parameters
    ----------
    i : int
        Id of grid cell
    k : int
        Id of fold
    estimator : BaseGradientBoosting
        The GBRT estimator
    params : dict
        The parameter settings for the grid cell.
    X_train : np.ndarray, shape=(n, m)
        The training data
    y_train : np.ndarray, shape=(n,)
        The training targets
    X_test : np.ndarray
        The test data
    y_test : np.ndarray
        The test targets
    """
    i, k, estimator, params, X_train, y_train, X_test, y_test = args

    estimator = clone(estimator)
    estimator.set_params(**params)

    t0 = time()
    estimator.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    scores = estimator.staged_score(X_test, y_test)
    test_time = time() - t0
    res = {'grid_cell': i, 'fold': k, 'params': params,
           'scores': map(float, scores.astype(np.float).tolist()),
           'train_time': train_time, 'test_time': test_time}

    get_mongodb().grid_search.insert(res, safe=True, continue_on_error=False)
    return res

def main(X, y):
    X = X
    y = y
    estimator = GradientBoostingClassifier()
    K = 2
    
    param_grid = {'n_estimators': [1000, 3000, 5000],
                  'min_samples_leaf': [7, 9, 13],
                  'max_depth': [4, 5, 6, 7],
                  'max_features': [100, 150, 250],
                  'learn_rate': [0.005, 0.02, 0.01],
                  }

    grid = ParameterGrid(param_grid)
    grid_size = sum(1 for params in grid)
    print("_" * 80)
    print("GridSearch")
    print("grid size: %d" % grid_size)
    print("num tasks: %d" % (K * grid_size))

    cv = KFold(X.shape[0], K, shuffle=True, random_state=0)

    # instantiate the tasks - K times the number of grid cells
    # FIXME use generator to limit memory consumption or do fancy
    # indexing in _parallel_grid_search.
    tasks = [(i, k, estimator, params, X[train], y[train], X[test], y[test])
             for i, params in enumerate(grid) for k, (train, test)
             in enumerate(cv)]

    # distribute tasks on ipcluster
    rc = parallel.Client()
    lview = rc.load_balanced_view()
    results = lview.map(_parallel_grid_search, tasks)

    # when fin run::
    summarize_results()

if __name__ == '__main__':
    df = pd.read_csv('../data/master_RushPassOnly.csv')
    y = df.pop('IsPass').values
    X = df.values

    main(X, y)