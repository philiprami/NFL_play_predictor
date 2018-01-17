import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.grid_search import GridSearchCV

# Load the dataset in with pandas
df = pd.read_csv('../data/master.csv')

# Limit the dataset to just rush or pass plays
mask_pass = (df['PlayType'] == 'PASS') | (df['IsPass'] == 1)
mask_rush = (df['PlayType'] == 'RUSH') | (df['IsRush'] == 1)
df = df[mask_pass | mask_rush]

# Make a numpy array called y containing the IsPass values
y = df.pop('IsPass').values

# Get dummies for categorical data (try label encoding)
categorical = ['OffenseTeam', 'DefenseTeam','Formation', 'Location', 'Last_play',
                'Surface', 'Weather_cat', 'Coach', 'Offensive_coordinator', 'SeasonYear',
                'Offensive_scheme', 'Defensive_coordinator', 'Defensive_alignment']

df = pd.get_dummies(df, columns=categorical)

# Drop columns that won't be trained on
drop_columns = ['GameId', 'GameDate', 'NextScore', 'Description', 'TeamWin',
                'Yards', 'PlayType', 'IsRush', 'IsIncomplete', 'IsTouchdown', 
                'PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed', 'IsMeasurement',
                'IsInterception', 'IsFumble', 'IsPenalty', 'IsTwoPointConversionScore',
                'IsTwoPointConversionSuccessful', 'RushDirection', 'YardLineFixed',
                'YardLineDirection', 'IsPenaltyAccepted', 'PenaltyTeam', 'IsNoPlay',
                'PenaltyType', 'PenaltyYards', 'Team1_Team2', 'Away_team', 'SeriesFirstDown',
                'Home_team', 'IsFieldGoal', 'IsSafety', 'Home_score', 'Away_score'
                ]

df = df.drop(drop_columns, axis=1)

# Make a 2 dimensional numpy array containing the feature data (everything except the labels)
X = df.values

# Use sklearn's train_test_split to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use sklearn's RandomForestClassifier to build a model of your data
rf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=45, max_features=None, oob_score=True, n_jobs=-1, verbose=2)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, verbose=2)
gb.fit(X_train, y_train)

# Use grid search to find the best parameters for the random forest classifier
# gb_parameters = {"n_estimators" : [100, 150],
#                  "learning_rate": [0.01],
#                  } 

# gb = GradientBoostingClassifier()
# cv_gb = GridSearchCV(gb, gb_parameters, cv=5, n_jobs=-1, verbose=2)
# cv_gb.fit(X_train, y_train)
# best_parameters = cv_gb.best_params_
# print best_parameters

# pickle the model to disk
rf_filename = '../data/rf_MVP.p'
gb_filename = '../data/gb_MVP.p'
pickle.dump(rf, open(rf_filename, 'wb'))
pickle.dump(gb, open(gb_filename, 'wb'))
  
# load the model from disk
rf_model = pickle.load(open(rf_filename, 'rb'))
gb_model = pickle.load(open(gb_filename, 'rb'))

# calculate accuracy, precision, recall, and feature importances
def return_model_scores(model, X_test, y_test, n_features):
        accuracy = model.score(X_test, y_test)
        feature_importances = np.argsort(rf.feature_importances_)
        y_predict = rf.predict(X_test)
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        confusion_matrix = confusion_matrix(y_test, y_predict)
        return score, list(df.columns[feature_importances[-1:-(n_features+1):-1]]), \
          precision, recall, confusion_matrix

accuracy, feature_importances, precision, recall, confusion_matrix = return_model_scores(rf, X_test, y_test, 20)

