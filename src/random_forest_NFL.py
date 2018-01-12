from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from roc import plot_roc

# Load the dataset in with pandas
df = pd.read_csv('../data/master.csv')

# Limit the dataset to just rush or pass plays
mask_pass = (df['PlayType'] == 'PASS') | (df['IsPass'] == 1)
mask_rush = (df['PlayType'] == 'RUSH') | (df['IsRush'] == 1)
df = df[mask_pass | mask_rush]

# Make a numpy array called y containing the IsPass values
y = df['IsPass'].values
y = df.pop('IsPass').values

# Get dummies for categorical data (try label encoding)
categorical = ['OffenseTeam', 'DefenseTeam','Formation', 'Location',
                'Surface', 'Weather_cat', 'Coach', 'Offensive_coordinator', 
                'Offensive_scheme', 'Defensive_coordinator', 'Defensive_alignment']

df = pd.get_dummies(df, columns=categorical)

# Drop columns that won't be trained on
drop_columns = ['GameId', 'GameDate', 'NextScore', 'Description', 'TeamWin',
                'Yards', 'PlayType', 'IsRush', 'IsIncomplete', 'IsTouchdown', 
                'PassType', 'IsSack', 'IsChallenge', 'IsChallengeReversed', 'IsMeasurement',
                'IsInterception', 'IsFumble', 'IsPenalty', 'IsTwoPointConversion',
                'IsTwoPointConversionSuccessful', 'RushDirection', 'YardLineFixed',
                'YardLineDirection', 'IsPenaltyAccepted', 'PenaltyTeam', 'IsNoPlay',
                'PenaltyType', 'PenaltyYards', 'Team1_Team2', 'Away_team',
                'Home_team', 'IsFieldGoal', 'IsSafety', 'Home_score', 'Away_score'
                ]

df = df.drop(drop_columns, axis=1)

# Make a 2 dimensional numpy array containing the feature data (everything except the labels)
X = df.values

# Use sklearn's train_test_split to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use sklearn's RandomForestClassifier to build a model of your data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# What is the accuracy score on the test data?
print "RF score:", rf.score(X_test, y_test)
print "GB score:", gb.score(X_test, y_test)
## answer: 0.9448441247

# 9. Draw a confusion matrix for the results
y_predict = rf.predict(X_test)
print "confusion matrix:"
print confusion_matrix(y_test, y_predict)
## answer:  716   6
##           40  72

# 10. What is the precision? Recall?
print "precision:", precision_score(y_test, y_predict)
print "recall:", recall_score(y_test, y_predict)
## precision: 0.923076923077
##    recall: 0.642857142857