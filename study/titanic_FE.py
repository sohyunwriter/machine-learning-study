/**
    ML STUDY 2(19.07.23) - titanic dataset preprocessing
    
    reference : 
    https://www.kaggle.com/yogi045/preprocess-and-predicting-using-random-forest
    https://cyc1am3n.github.io/2018/10/09/my-first-kaggle-competition_titanic.html
    
    @sohyunwriter (brightcattle@gmail.com)
**/

/**This is just for interim check for study, so below code is not working. I'll reupload full code asap**/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # set seaborn default for plots
from data_analyzer import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import numpy as np

# feature engineering
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
train.head()

# data dictionary
# Survived : survival (0 = No, 1 = Yes)
# Pclass : ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# Sex
# Age : age in years
# SibSp : # of siblings/spouses aboard the Titanic
# Parch : # of parents/children aboard the Titanic
# Ticket : Ticket number
# Fare : Passenger fare
# Cabin : Cabin number
# Embarked : Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# get info of train data and test data
print('train data shape : ', train.shape)
print('test data shape : ', test.shape)
print('---- train info ----')
print(train.info())
print('---- test info ----')
print(test.info())

# 1) feature - Sex
#pie_chart(train, 'Sex')
# male number > female number
# male's survived rate < female's survived rate

# 2) feature - Pclass
#pie_chart(train, 'Pclass')
# Plcass 3 number is the largest
# Plcass index lower -> survived higher

# 3) feature - Embarked
#pie_chart(train, 'Embarked')
# S number is the largest
# C : survived rate > dead rate
# S : survived rate < dead rate
# Q : survived rate < dead rate

# 4) feature - SibSp
#bar_chart(train, 'SibSp')

# merge train, test dataset
#train_and_test = pd.concat([train, test])
#print(train_and_test)
train_and_test = [train, test]  # list

## data preprocessing
# - extract whether Master/Misss/Mr/Mrs/Others from 'Name' feature   <feature engineering 1>

for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.') # blank + ([A-Za-z]+) + .
train.head(5)

#print(pd.crosstab(train['Title'], train['Sex']))

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona','Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Others')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)

#print(train_and_test)

# - Sex Feature
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)

# - Embarked Feature
print(train.Embarked.value_counts(dropna=False))
'''
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
'''

for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)

# - Age Feature
########## age feature is not working
for dataset in train_and_test:
    sex = dataset['Sex']
    title = dataset['Title']

    try:
        avg = train[train['Title'] == title][train['Age'].notnull()].mean()
        std = train[train['Title'] == title][train['Age'].notnull()].std()
       # print("sssssssssssssss")
        print(avg['Age'], std['Age'])
        rand_1 = np.random.randint(avg['Age'] - std['Age'], avg['Age'] + std['Age'])

    except:
        avg = train[train['Sex'] == sex][train['Age'].notnull()].mean()
        std = train[train['Sex'] == sex][train['Age'].notnull()].std()
        print(avg['Age'], std['Age'])
        rand_1 = np.random.randint(avg['Age'] - std['Age'], avg['Age'] + std['Age'])

# - Fare Feature
print(train.Fare.isnull().sum()) # 0
print(test.Fare.isnull().sum()) # 1
print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"]) # isnull data is in pclass 3

for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(train[train['Pclass'] == 3]["Fare"].mean())
print(train[train['Pclass'] == 3]["Fare"].mean())
# drop unused columns

# - SibSp Feature & Parch Feature (Famliy)   <Feature Engineering 2>
for dataset in train_and_test:
    dataset['Family'] = dataset['Parch'] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)

print(train.columns)
print(test.columns)

################################################################################
## drop columns
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)

# one-hot encoding for categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.columns)

train_label = train['Survived']
train_data = train.drop('Survived', axis = 1)

'''
x_train = train.iloc[:, 0]
y_train = train.iloc[:, 1:]
print(x_train, y_train)

# train x, train y
columns = ['Sex', 'Embarked', 'Title', 'Family']
Y_columns = ['Survived']

# get dummies columns from categorical value in dataframe in both train and test ??
y_train, x_train = exCategories(train, columns, Y_columns)
y_test, x_test = exCategories(test, columns, Y_columns)
print(x_train, y_train)

# modeling
logreg = LogisticRegression(random_state = 7, n_jobs = -1)
print('7-fold cross validation:\n')
scores = cross_val_score(x_train, y_train.Survived, cv=5, scoring="accuracy")
print("Accuracy: %0.4f (+/- %0.4f) " % (scores.mean(), scores.std()))
'''

train_data, train_label = shuffle(train_data, train_label, random_state=5)
model = LogisticRegression()
model.fit(train_data, train_label)
prediction = model.predict(test)
accuracy = round(model.score(train_data, train_label) * 100, 2)
print("Accuracy: ", accuracy, "%")
