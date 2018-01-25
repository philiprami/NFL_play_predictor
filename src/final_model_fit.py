import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# After much grid search, the final hyperparameters:
# {'learning_rate': 0.02, 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 8000}

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return dataframe, X_train, X_test, y_train, y_test

def fit_model(X_train, y_train):
    model = GradientBoostingClassifier(learning_rate=0.02, 
                                       max_depth=4, 
                                       max_features='sqrt',
                                       n_estimators=8000,
                                       verbose=2
                                       )
    model.fit(X_train, y_train)
    return model

def return_model_scores(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    baseline = y_test.mean()
    y_predict = model.predict(X_test)
    class_report = classification_report(y_test, y_predict)
    conf_matrix = confusion_matrix(y_test, y_predict)
    return accuracy, baseline, class_report, conf_matrix

def get_sorted_features(df, model, n_features):
    feature_importance = {}
    for label, importance in zip(df.columns, model.feature_importances_):
        feature_importance[label] = importance

    return sorted(feature_importance.items(), key=lambda x: (-x[1]))[:n_features + 1]

def pickle_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    df = pd.read_csv('../data/master.csv')
    df, X_train, X_test, y_train, y_test = process_frame(df)
    model = fit_model(X_train, y_train)
    accuracy, baseline, class_report, conf_matrix = \
      return_model_scores(model, X_test, y_test)
    important_features = get_sorted_features(df, model, 10)
    print best_params
    print accuracy
    print baseline
    print important_features
    print class_report
    print conf_matrix
    pickle_model('../models/gb_final.p', model)
    pickle_model('../models/X_test.p', X_test)
    pickle_model('../models/y_test.p', y_test)
    pickle_model('../models/X_train.p', X_train)
    pickle_model('../models/y_train.p', y_train)