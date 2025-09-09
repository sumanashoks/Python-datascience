#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('/Users/apple/Downloads/OnlineRetail.csv',encoding='latin-1')


# In[4]:


data


# In[5]:


from sklearn.model_selection import train_test_split
X=data[['Quantity']]
y=data['UnitPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[6]:


import numpy as np

X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
error_rate = []
for i in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    


# In[ ]:




