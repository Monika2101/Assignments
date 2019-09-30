#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_iris
dataset=load_iris()


# In[3]:


X=dataset.data
y=dataset.target


# In[4]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[6]:


knn.score(X_test,y_test)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()
dtf.fit(X_train, y_train)


# In[8]:


y_pred=dtf.predict(X_test)


# In[9]:


dtf.score(X_train,y_train)

