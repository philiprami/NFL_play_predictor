import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.grid_search import GridSearchCV

# Load the dataset in with pandas
df = pd.read_csv('../data/master.csv')

# Limit the dataset to just rush or pass plays
mask_pass = df['PlayType'] == 'PASS'
mask_rush = (df['PlayType'] == 'RUSH') | (df['IsRush'] == 1)
df = df[mask_pass | mask_rush]

# Make a numpy array called y containing the IsPass values
y = df.pop('IsPass').values

# Get dummies for categorical data (try label encoding)
categorical = ['OffenseTeam', 'DefenseTeam', 'Formation', 'Location', 'Month', 'Quarter',
                'Surface', 'Weather_cat', 'Coach', 'Offensive_coordinator', 'SeasonYear',
                'Offensive_scheme', 'Defensive_coordinator', 'Defensive_alignment', 'Down'
                ]

df = pd.get_dummies(df, columns=categorical)

# Drop columns that won't be trained on
drop_columns = ['GameId', 'GameDate', 'NextScore', 'Description', 'TeamWin', 'Minute', 'Second',
                'Yards', 'PlayType', 'IsRush', 'IsIncomplete', 'IsTouchdown', 'Two_plays_ago',
                'PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed', 'IsMeasurement',
                'IsInterception', 'IsFumble', 'IsPenalty', 'IsTwoPointConversionScore',
                'IsTwoPointConversionSuccessful', 'RushDirection', 'YardLineFixed', 'IsExtraPoint',
                'YardLineDirection', 'IsPenaltyAccepted', 'PenaltyTeam', 'IsNoPlay',
                'PenaltyType', 'PenaltyYards', 'Team1_Team2', 'Away_team', 'SeriesFirstDown',
                'Home_team', 'IsFieldGoal', 'IsSafety', 'Home_score', 'Away_score', 'Last_play',
                ]

df = df.drop(drop_columns, axis=1)

# Make a 2 dimensional numpy array containing the feature data (everything except the labels)
X = df.values

# Use sklearn's train_test_split to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Use sklearn's tree models to build MVPs
rf = RandomForestClassifier(n_estimators=2000, min_samples_leaf=45, max_features=None, oob_score=True, n_jobs=-1, verbose=2)
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier(max_features='sqrt',n_estimators=4000,learning_rate=0.05, max_depth=4, verbose=2)
gb.fit(X_train, y_train)

ada = AdaBoostClassifier(learning_rate=0.01, n_estimators=1000)
ada.fit(X_train, y_train)

# pickle the model to disk
rf_filename = '../models/rf_MVP.p'
gb_filename = '../models/gb_MVP.p'
ada_filename = '../models/ada_MVP.p'
pickle.dump(rf, open(rf_filename, 'wb'))
pickle.dump(gb, open(gb_filename, 'wb'))
pickle.dump(ada, open(ada_filename, 'wb'))
  
# load the model from disk
# rf_model = pickle.load(open(rf_filename, 'rb'))
# gb_model = pickle.load(open(gb_filename, 'rb'))
# ada_model = pickle.load(open(ada_filename, 'rb'))

# calculate accuracy, precision, recall, f1 score, and feature importances
def return_model_scores(df, model, X_test, y_test, n_features):
    accuracy = model.score(X_test, y_test)
    baseline = y_test.mean()
    feature_importances = np.argsort(model.feature_importances_)
    feature_list = list(df.columns[feature_importances[-1:-(n_features+1):-1]])
    y_predict = model.predict(X_test)
    class_report = classification_report(y_test, y_predict)
    conf_matrix = confusion_matrix(y_test, y_predict)
    return accuracy, baseline, feature_list, class_report, conf_matrix

models = [rf, gb, ada]
for model in models:
        print return_model_scores(model, X_test, y_test, 10), '\n'
