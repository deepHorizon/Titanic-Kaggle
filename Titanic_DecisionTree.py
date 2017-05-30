# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:44:48 2017

@author: Shaurya Rawat
"""
#data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from django.conf.urls import url,include

#data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#Read the train and test dataset
train=pd.read_csv("D:\\Kaggle\\Titanic Disaster\\train.csv")
test=pd.read_csv("D:\\Kaggle\\Titanic Disaster\\test.csv")
combine=[train,test]#combine the test and train dataset

train.columns.values #check the columns in the train dataset
#check the head and tail of the dataset
train.head()
train.tail()
train.info()
test.info()
train.describe() #for numerical features
train.describe(include==["O"]) #for categorical features
#Analyze by pivoting features
#To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
#1) Pclass : We observe significant correlation (>0.5) among Pclass=1 and Survived . We decide to include this feature in our model.
#2) Sex : We confirm the observation during problem definition that Sex=female had very high survival rate at 74%.
#3) SibSp and Parch: These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features.

train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'],
                                  as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'],
                                  as_index=False).mean().sort_values(by='Survived', ascending=False)

train[['Parch', 'Survived']].groupby(['Parch'],
                                  as_index=False).mean().sort_values(by='Survived', ascending=False)


#Data Preparation

#Categorical variables need to  be transformed into numerical variables
#sex will be encoded into 0 or 1 based on male or female
for dataset in combine:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)  
train.head(5)
#the sex is now shown as a binary variable 0:male and 1:female

#dropping out some features
print("Before",train.shape,test.shape,combine[0].shape,combine[1].shape)
train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
print("After",train.shape,test.shape,combine[0].shape,combine[1].shape)

#creating new feature extracting from existing
#we create a new feature Title of all the passengers
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])


for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()

#now we  map the different titles in the dataset to a number
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
train.head(5)

train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name','PassengerId'],axis=1)
train.shape,test.shape

##FILLING MISSING VALUES
#fill missing values of age with mean of age
train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())

#fill missing values of fare with the mean of fare
train['Fare']=train['Fare'].fillna(train['Fare'].mean())
test['Fare']=test['Fare'].fillna(test['Fare'].mean())

#now we check if any attribute has any missing values
train.isnull().any()
#this shows that fare and age dont have missing values anymore, but embarked is showing as True so we fix it

#now we find the port from which most passengers embarked
freq_port=train['Embarked'].mode()[0] 
freq_port #'S': South Hampton is the most frequent embarked port
#  as south hampton is the most frequent port. we put the missing values in the embarked in the na values in embarked
for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].fillna(freq_port)
train[['Embarked','Survived']].groupby('Embarked',as_index=False).mean().sort_values(by='Survived',ascending=False)
#after running the above command to check the survival rate of people from different ports. 
# we see that people embarking from 'C' have more survival rate and least is 'S'

#now we map the 3 ports as 1,2,3 as we had done for title
for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

##MODEL PREDICT AND SOLVE
#We are gonna use the following methods of machine learning on our dataset now
#Logistic Regression
#KNN
#Support Vector Machines SVM
#Naive Bayes Classifier
#Decision Tree
#Random Forest
#Perceptron
#Artificial Neural Network
#Relevance Vector Machine RVM

#we create xtrain ytrain and xtest
X_train=train.drop('Survived',axis=1)
Y_train=train['Survived']
X_test=test

#check the shape
X_train.shape,Y_train.shape,X_test.shape
#train is still showing the embarked ports as s c and q so we map it
train['Embarked']=train['Embarked'].fillna(freq_port)
test['Embarked']=test['Embarked'].fillna(freq_port)
train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

##LOGISTIC REGRESSION
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
acc_log=round(logreg.score(X_train,Y_train)*100,1)
acc_log
#Score= 81.4 %

##KNN or K-nearest-neighbors
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
acc_knn=round(knn.score(X_train,Y_train)*100,1)
acc_knn
#Score= 84.8 %

##Support Vector Machines
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
acc_svc=round(svc.score(X_train,Y_train)*100,1)
acc_svc
#Score= 91.1 %

##Gaussian Naive Bayes
gaussian=GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred=gaussian.predict(X_test)
acc_gaussian=round(gaussian.score(X_train,Y_train)*100,1)
acc_gaussian
#Score= 79.9 %

##Linear SVC
linear_svc=LinearSVC()
linear_svc.fit(X_train,Y_train)
Y_pred=linear_svc.predict(X_test)
acc_linear_svc=round(linear_svc.score(X_train,Y_train)*100,1)
acc_linear_svc

##Decision Tree Classifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_pred=decision_tree.predict(X_test)
acc_decision_tree=round(decision_tree.score(X_train,Y_train)*100,2)
acc_decision_tree
#Score: 98.4 %

##Random Forest Classifier
random_forest=RandomForestClassifier()
random_forest.fit(X_train,Y_train)
Y_pred=random_forest.predict(X_test)
acc_random_forest=round(random_forest.score(X_train,Y_train)*100,2)
acc_random_forest
#Score: 96.3 %






