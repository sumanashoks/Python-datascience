#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


print('missing values check')
missing_values=df.isnull().sum()
print(missing_values)


# In[9]:


df_num=df.select_dtypes(np.number)


# 

# In[10]:


df_num


# In[19]:


df


# In[23]:


df.loc[[1,2]]


# In[25]:


df.loc[1:5,['mean radius','mean texture']]


# In[30]:


#df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.rename(columns={'mean radius':'ROCKY_BHAI'},inplace=True)


# In[34]:


df.loc[0:1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


correlation_matrix=df.corr()
sns.heatmap(correlation_matrix,annot=True)


# In[12]:


df.head()


# In[13]:


sns.jointplot(data=df,x='worst symmetry',y='target')


# In[14]:


sns.countplot(x=df['target'])


# In[15]:


from sklearn.model_selection import  train_test_split
X= df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=720)
X_train


# In[16]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[17]:


error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1, 20), error_rate)
plt.title('Cross-Validation Scores for Different K Values')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate')
plt.show()


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix
#Build the Model with 3 as K
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

# Predict and Evaluate the Model
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:





# In[ ]:




