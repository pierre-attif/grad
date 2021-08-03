from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 
import pandas as pd 
import os 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

#Load the data 
from google.colab import files 
uploaded = files.upload()

df = pd.read_csv('SNP.CSV')
list_drop = ['Type']
df.drop(list_drop , axis=1 , inplace=True)
#Encoding the sequence Columns
encoded_columns = pd.get_dummies(df['Gene'])
#overriding the new dataset
df = df.join(encoded_columns).drop('Gene' , axis = 1 )
df.head(1)

df.isna().sum()

#stoped here 
#Visualize the count
sns.countplot(df['POSITION'] , label = 'count')

#Get the correlation of the columns 
df.iloc[:,1 : 30].corr()

#Visualize the correlation 
plt.figure(figsize=(8,8))
sns.heatmap(df.iloc[:,1 : 20].corr() , annot = True , fmt = '.0%' )

#Split the data to dependant(X) , and independant data cells(Y) 
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#Split the data to 25% testing and 75% training 
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size = 0.25 , random_state = 0)

#Scale the data -  Feature scaling  
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def models(X_train,Y_train):

   #Logistic regression
  from sklearn.linear_model import LogisticRegression 
  log = LogisticRegression(random_state=0)
  log.fit(X_train,Y_train)


  #Random Estimators 
  from sklearn.ensemble import RandomForestRegressor
  forest = RandomForestRegressor(n_estimators = 10 , random_state=0)
  forest.fit(X_train,Y_train) 

  ##Decesion Tree 
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy' , random_state=0)
  tree.fit(X_train,Y_train) 
  
  print('[0] Logisitc regresson accuracy is' , log.score(X_train,Y_train) * 100)
  print('[2] Random forest accuracy is ' ,forest.score(X_train,Y_train) * 100) 
  print('[1] Decesion tree accuracy is ' , tree.score(X_train,Y_train) * 100)
  
  return log, forest , tree

model = models(X_train,Y_train)

#model's accuracy on test dataset
model = models(X_test,Y_test)
