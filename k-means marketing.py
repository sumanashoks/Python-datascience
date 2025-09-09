#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/KMeans/K Means Assignment/Marketing_data.csv')


# In[3]:


data


# In[4]:


data


# In[5]:


data.dtypes


# In[6]:


data.columns


# In[7]:


data.head(n=10)


# In[8]:


data.describe()


# In[9]:


data[data['ONEOFF_PURCHASES']==40761.250000]


# In[10]:


data[data['BALANCE']>15000]


# In[11]:


print('missing values')
missing_values= data.isnull().sum()
print(missing_values)


# In[12]:


data.isnull().sum()


# # THIS GRAPH GIVES GRAPHICAL REPRESENTATION OF NULL VALUES

# In[13]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = True, cmap="Blues")


# In[14]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# # I'M REMOVING NULL VALUES.

# In[15]:


data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(), inplace=True)


# In[16]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = True, cmap="Blues")


# In[17]:


data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)


# In[18]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# In[19]:


data.describe()


# In[20]:


data.info()


# In[21]:


data.isnull().sum()


# # I'M CHECKING DUPLICACTE VALUES

# In[22]:


data.duplicated().sum()


# In[23]:


data.drop(['CUST_ID'], axis=1,inplace=True)


# In[24]:


data


# In[25]:


plt.figure(figsize=(10,50))
for i in range(len(data.columns)):
  plt.subplot(17, 1, i+1)
  sns.distplot(data[data.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
  plt.title(data.columns[i])

plt.tight_layout()


# In[26]:


scores_1 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(data)
  scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()


# In[27]:


sse={}

#For Loop to capture the Inertia

for k in range(1, 20):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_

print(sse)
#Store the no. of groups and the error as separate lists
groups=list(sse.keys())
error=list(sse.values())

#Club the lists as a dataframe
error_data= pd.DataFrame(
    {'groups': groups,
     'error': error
    })
error_data.head()
sns.pointplot(x="groups", y="error", data=error_data)


# In[28]:


data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)


# In[29]:


sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap='Blues')


# In[30]:


scores_1 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(data)
  scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()


# In[31]:


sse={}

#For Loop to capture the Inertia

for k in range(1, 20):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_

print(sse)
#Store the no. of groups and the error as separate lists
groups=list(sse.keys())
error=list(sse.values())

#Club the lists as a dataframe
error_data= pd.DataFrame(
    {'groups': groups,
     'error': error
    })
error_data.head()

sns.pointplot(x="groups", y="error", data=error_data)
plt.grid()


# In[32]:


kmeans=KMeans(n_clusters=4)
kmeans.fit(data)


# In[33]:


kmeans.cluster_centers_.shape


# In[34]:


kmeans.cluster_centers_


# In[35]:


data['cluster']=kmeans.labels_


# In[36]:


data


# In[37]:


data[data['cluster']==0]


# In[38]:


data[data['cluster']==1]


# In[39]:


data.groupby('cluster').describe()


# In[40]:


data.groupby('cluster')


# In[41]:


data.groupby('cluster').describe()


# In[42]:


# Assuming 'data' is your DataFrame
grouped_data = data.groupby('cluster').describe()

# Specify the path where you want to save the CSV file
csv_path = '/Users/apple/Desktop/python/KMeans/file.csv'

# Export the grouped data to CSV
grouped_data.to_csv(csv_path)


# In[43]:


import pandas as pd

# Assuming you have your data in a DataFrame called 'data'
# For example:
# data = pd.read_csv('/path/to/your/data.csv')

# Saving the DataFrame to a new CSV file
data.to_csv('/Users/apple/Desktop/python/KMeans/exported_data.csv', index=False)


# In[45]:


data[data['cluster']==1]


# In[46]:


data[data['cluster']==0]


# In[47]:


data[data['cluster']==2]


# In[48]:


data[data['cluster']==3]


# In[52]:


sns.countplot(data=data, x='cluster')


# In[ ]:





# In[55]:


from sklearn.model_selection import train_test_split

#Split Dataset
X = data.drop(['cluster'],axis=1)
y= data[['cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3)


# In[59]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[60]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[ ]:




