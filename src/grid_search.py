import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def process_frame(dataframe):
    mask_pass = dataframe['PlayType'] == 'PASS'
    mask_rush = (dataframe['PlayType'] == 'RUSH') | (dataframe['IsRush'] == 1)
    dataframe = dataframe[mask_pass | mask_rush]
    y = dataframe.pop('IsPass').values
    categorical = ['OffenseTeam', 'DefenseTeam', 'Formation', 
                   'Quarter', 'Surface', 'Weather_cat',
                   'Offensive_scheme', 'Defensive_alignment', 
                   'Down','Defensive_coordinator','Coach', 
                   'Offensive_coordinator', 'Location', 'Month'
                   ]

    dataframe = pd.get_dummies(dataframe, columns=categorical)
    drop_columns = ['GameId', 'GameDate', 'NextScore', 'Description', 'TeamWin', 'Minute', 
                    'Second', 'Yards', 'PlayType', 'IsRush', 'IsIncomplete', 'IsTouchdown', 
                    'Two_plays_ago', 'PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed', 
                    'IsMeasurement', 'IsInterception', 'IsFumble', 'IsPenalty', 
                    'IsTwoPointConversionScore', 'IsTwoPointConversionSuccessful', 'RushDirection', 
                    'YardLineFixed', 'IsExtraPoint', 'YardLineDirection', 'IsPenaltyAccepted', 
                    'PenaltyTeam', 'IsNoPlay', 'SeasonYear', 'PenaltyType', 'PenaltyYards', 
                    'Team1_Team2', 'Away_team', 'SeriesFirstDown', 'Home_team', 'IsFieldGoal', 
                    'IsSafety', 'Home_score', 'Away_score', 'Last_play'
                    ]

    dataframe = dataframe.drop(drop_columns, axis=1)
    X = dataframe.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return dataframe, X_train, X_test, y_train, y_test

def gs_fit_model(model, X_train, y_train):

    grid = {'max_features': ['sqrt'],
            'n_estimators': [4000],
            'learning_rate': [0.05],
            'max_depth' : [3, 4]
            } 
    
    gridsearch = GridSearchCV(model, grid, verbose=2, cv=4, n_jobs=-1)
    gridsearch.fit(X_train, y_train)
    return gridsearch.best_params_, gridsearch.best_estimator_

def return_model_scores(df, model, X_test, y_test, n_features):
    accuracy = model.score(X_test, y_test)
    baseline = y_test.mean()
    feature_importances = np.argsort(model.feature_importances_)
    feature_list = list(df.columns[feature_importances[-1:-(n_features+1):-1]])
    y_predict = model.predict(X_test)
    class_report = classification_report(y_test, y_predict)
    conf_matrix = confusion_matrix(y_test, y_predict)
    return accuracy, baseline, feature_list, class_report, conf_matrix
    # get feature_list function from data exploration

def pickle_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def load_pickled_model(filename):
    return pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    df = pd.read_csv('../data/master.csv')
    df, X_train, X_test, y_train, y_test = process_frame(df)
    # define model here... GB, RF, ADA
    model = GradientBoostingClassifier(verbose=2)
    best_params, best_model = gs_fit_model(model, X_train, y_train)
    accuracy, baseline, important_features, class_report, conf_matrix = \
      return_model_scores(df, best_model, X_test, y_test, 20)
    print best_params
    print accuracy
    print baseline
    print feature_importances
    print class_report
    print conf_matrix
    pickle_model('../models/gb_1.19.6.p', best_model)

