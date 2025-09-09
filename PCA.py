#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('/Users/apple/Desktop/python/KMeans/K Means Assignment/Marketing_data.csv')


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(subset=['CREDIT_LIMIT'],inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df['TENURE'].value_counts(normalize=True)*100


# In[10]:


df.describe()


# In[11]:


df[df['PURCHASES']==49039.570000]


# In[12]:


df.drop(['CUST_ID'],inplace=True,axis=1)


# In[16]:


#df.isnull().sum()
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(),inplace=True)


# In[17]:


df


# In[18]:


df.isnull().sum()


# In[19]:


from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scaled_df = scalar.fit_transform(df)


# In[20]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
pca_df


# In[21]:


df.shape


# In[22]:


scaled_df.shape


# In[23]:


pca_df.shape


# # PCA GRAPH

# In[27]:


plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PCA1"], pca_df["PCA2"],cmap='plasma')
plt.title('PCA Components Scatter Plot')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


# # BEFORE SCALLING

# In[25]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df.values.flatten(), bins=50, kde=True, color='blue')
plt.title('Before Scaling')
plt.xlabel('Values')
plt.ylabel('Frequency')


# # AFTER SCALLING

# In[26]:


plt.subplot(1, 2, 2)
sns.histplot(scaled_df.flatten(), bins=50, kde=True, color='green')
plt.title('After Scaling')
plt.xlabel('Scaled Values')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[2]:


# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='rbf', C=1)
# Train the SVM classifier on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the testing set
predictions = svm_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of SVM on the test set: {accuracy:.2f}")


# In[ ]:




