# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:53:30 2017

@author: Shaurya Rawat
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

#algos
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#helpers
from sklearn.preprocessing import Imputer,Normalizer,scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure the visualizations
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']=8,6

# Setup helper functions

def plot_histogram(df,variables,n_rows,n_cols):
       fig=plt.figure(figsize=(16,12))
       for i, varname in enumerate(variables):
              ax=fig.add_subplot(n_rows,n_cols,i+1)
              df[var_name].hist(bins=10,ax=ax)
              ax.set_title('Skew:'+str(round(float(df[var_name].skew()),)))
              ax.set_xticklabels([],visible=False)
              ax.set_yticklabels([],visible=False)
       fig.tight_layout()
       plt.show()

def plot_distribution(df,var,target,**kwargs):
       row=kwargs.get('row',None)
       col=kwargs.get('col',None)
       facet=sns.FacetGrid(df,hue=target,aspect=4,row=row,col=col)
       facet.map(sns.kdeplot,var,shade=True)
       facet.set(xlim=(0,df[var].max()))
       facet.add_legend()

def plot_categories(df,cat,target,**kwargs):
       row=kwargs.get('row',None)
       col=kwargs.get('col',None)
       facet=sns.FacetGrid(df,row=row,col=col)
       facet.map(sns.barplot,cat,target)
       facet.add_legend()
       
def plot_correlation_map(df):
       corr=df.corr()
       _,ax=plt.subplots(figsize=(12,10))
       cmap=sns.diverging_palette(220,10,as_cmap=True)
       _=sns.heatmap(corr,cmap=cmap,square=True,cbar_kws={'shrink':.9},ax=ax,annot=True,annot_kws={'fontsize':12})

def describe_more(df):
       var=[];l=[];t=[]
       for x in df:
              var.append(x)
              l.append(len(pd.value_counts(df[x])))
              t.append(df[x].dtypes)
       levels=pd.DataFrame({'Variable':var,'Levels':1,'Datatype':t})
       levels.sort_values(by='Levels',inplace=True)
       return levels

def plot_variable_importance(X,y):
       tree=DecisionTreeClassifier(random_state=99)
       tree.fit(X,y)
       plot_model_var_imp(tree,X,y)
       
def plot_model_var_imp(model,X,y):
       imp=pd.DataFrame(model.feature_importances_,columns=['Importance'],index=X.columns)
       imp=imp.sort_values(['Importance'],ascending=True)
       imp[:10].plot(kind='barh')
       print(model.score(X,y))
       
## read the data into pandas dataframe
train=pd.read_csv("D:\\Kaggle\\Titanic Disaster\\train.csv")  
test=pd.read_csv("D:\\Kaggle\\Titanic Disaster\\test.csv")    
full=train.append(test,ignore_index=True)
titanic=full[:891]

del train,test       


titanic.head()

#describe
titanic.describe()

#correlation
plot_correlation_map(titanic)

# plot distributions of age and survival
plot_distribution(titanic,var='Age',target='Survived',row='Sex')

# fare
plot_distribution(titanic,var='Fare',target='Survived',row='Sex')

# embarked and survived
plot_categories(titanic,cat='Embarked',target='Survived')

# survival rate of Sex
plot_categories(titanic,cat='Sex',target='Survived')

# survival rate by Pclass
plot_categories(titanic,cat='Pclass',target='Survived')

# survival rate by sibsp
plot_categories(titanic,cat='SibSp',target='Survived')

# survival rate by parch
plot_categories(titanic,cat='Parch',target='Survived')    

## Data preparation
#categorical values need to be changed to numerical values(transformation)

# transform sex into binary variable 
sex=pd.Series(np.where(full.Sex=='male',1,0),name='Sex')

#create a new value for every unique value of embarked
embarked=pd.get_dummies(full.Embarked,prefix='Embarked')
embarked.head()
# create a new variable for every value of Pclass
pclass=pd.get_dummies(full.Pclass,prefix='Pclass')
pclass.head()

## filling missing values
imputed=pd.DataFrame()
imputed['Age']=full.Age.fillna(full.Age.mean())
imputed['Fare']=full.Fare.fillna(full.Fare.mean())
imputed.head()

## feature engineering
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
#title = pd.concat( [ title , titles_dummies ] , axis = 1 )

title.head()

cabin = pd.DataFrame()

# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

cabin.head()       

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
ticket.head()     
       
family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()      

# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket

full_X = pd.concat( [ imputed , embarked , cabin , sex ] , axis=1 )
full_X.head()  

# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)

plot_variable_importance(train_X, train_y)


model = RandomForestClassifier(n_estimators=100)
model = SVC()
model = GradientBoostingClassifier()
model = KNeighborsClassifier(n_neighbors = 3)
model = GaussianNB()
model = LogisticRegression()
model.fit( train_X , train_y )
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))





 
       
       
       
       
       
       
       
       
       
       
       
       