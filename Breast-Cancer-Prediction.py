#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Approach for Breast Cancer Prediction 
# 

# In[8]:


#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[9]:


#reading data from file
df=pd.read_csv("./D/data.csv")


# In[10]:


df.info()


# In[11]:


#return all null values of columns
df.isna().sum()


# In[12]:


#return the size of dataset
df.shape


# In[13]:


#remove column
df=df.dropna(axis=1)


# In[14]:


#size of dataset after removing the column
df.shape


# In[15]:


#describe the dataset
df.describe()


# In[33]:


#get the count of malignant and benign cells
df['diagnosis'].value_counts()


# In[34]:


sns.countplot(x=df['diagnosis'],label="count")


# In[35]:


#Label Encoding(Covert the value of M and B into 1 and 0)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)


# In[36]:


df.head()


# In[37]:


sns.pairplot(df.iloc[:,1:5], hue="diagnosis")


# In[38]:


#get the correlation
df.iloc[:,1:32].corr()


# In[39]:


#visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:10].corr(),annot=True,fmt=".0%")


# In[40]:


#split the dataset into dependent(X) and independent(Y) datasets
X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values


# In[41]:


#splitting the data into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[42]:


#feature scaling
from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# In[54]:


#models/Algo

def models(X_train,Y_train):

    #    Logistic regression
    
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    #    Decision Tree
    
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)
    
    #    Random Forest
    
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    
    # SVM
    
    from sklearn import svm
    clf=svm.SVC(kernel='linear')
    clf.fit(X_train,Y_train)
    
    # K-NN
    
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,Y_train)
    
    
    
    print('[0]Logistic Regression Accuracy:',log.score(X_train,Y_train))
    print('[1]Decision Tree Accuracy:',tree.score(X_train,Y_train))
    print('[2]Random Forest Accuracy:',forest.score(X_train,Y_train))
    print('[3]Support Vector Machine Accuracy:',clf.score(X_train,Y_train))
    print('[4]KNN Accuracy:',neigh.score(X_train,Y_train))
    
    
   
    
    return log,tree,forest,clf,neigh


# In[55]:


model=models(X_train,Y_train)


# In[52]:


#testing the models/result
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print("Accuracy Score:",accuracy_score(Y_test,model[i].predict(X_test)))


# In[53]:


#prediction of random-forest
pred=model[2].predict(X_test)
print('Predicted Values:')
print(pred)
print('Actual Values:')
print(Y_test)


# In[47]:


from joblib import dump
dump(model[2],"Cancer_Prediction.joblib")


# In[48]:


df.head()

