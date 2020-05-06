# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:57:21 2019

@author: HP
"""
import pandas as pd

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix
#displayd data
from IPython.display import display
data = pd.read_csv('machinefoot.csv')

# Preview data.
display(data.head())
n_matches = data.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = data.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(data[data.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print ("Total number of matches:")
print (win_rate)

scatter_matrix(data[['FTHG','FTAG','HTR','HTHG','FTR','HS']], figsize=(10,10))
X_all = data.drop(['FTR'],1)
y_all = data['FTR']
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['FTHG','FTAG','HTR','HTHG','HS']]
for col in cols:
    X_all[col] = scale(X_all[col])
    X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output
X_all = preprocess_features(X_all)
display(X_all.head())
from sklearn.cross_validation import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 50,
                                                    random_state = 2,
                                                    stratify = y_all)
display(X_train.head())
from time import time 

from sklearn.metrics import f1_score
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
dataset=pd.read_csv('C:/Users/HP/Desktop/today1.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
label_code=LabelEncoder()
x[:,3]=label_code.fit_transform(x[:,3])
from sklearn.preprocessing import OneHotEncoder
hot_encode=OneHotEncoder(categorical_features=[3])
x=hot_encode.fit_transform(x).toarray()
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values=0,strategy="mean", axis=0)
x=imputer.fit_tranform(x)


    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    
    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))
def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    #print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print (f1 , acc)
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print (f1 , acc)
    clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print (" ")
train_predict(clf_B, X_train, y_train, X_test, y_test)
print (" ")
train_predict(clf_C, X_train, y_train, X_test, y_test)

