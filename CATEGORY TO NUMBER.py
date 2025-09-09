#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[2]:


ad_data=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[3]:


ad_data.describe()


# In[4]:


ad_data.info()


# In[5]:


print('missing values check')
missing_values=ad_data.isnull().sum()
print(missing_values)


# In[6]:


X


# In[7]:


from sklearn.model_selection import  train_test_split
X= df_feat
y=ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=720)


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']])
scaler_features=scaler.transform(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']])
df_feat=pd.DataFrame(scaler_features,columns=['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male'])


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
error_rate=[]
for i in range (1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.plot(range(1,20),error_rate)


# In[10]:


from sklearn.metrics import classification_report,confusion_matrix
#Build the Model with 2 as K
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[11]:


ad_data.dtypes


# In[ ]:


y


# In[ ]:


X


# In[ ]:


px.histogram(ad_data,x='Age',y='Daily Time Spent on Site')


# In[ ]:





# In[16]:


px.bar(ad_data,x='Age',y='Daily Time Spent on Site')


# In[17]:


px.line(ad_data,x='Age',y='Daily Time Spent on Site')


# In[18]:


px.scatter(ad_data,x='Age',y='Daily Time Spent on Site')


# In[19]:


px.pie(ad_data,values='Age')


# In[20]:


px.box(ad_data,x='Age',y='Daily Time Spent on Site')


# # YOU CAN USE HEAT MAP FOR ALL THE NUMERIC COLUMNS

# In[16]:


numeric_columns=ad_data.select_dtypes(include='number')
correlation_matrix=numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True)


# # YOU CAN USE HEAT MAP  FOR PARTICULAR COLUMNS 
# 

# In[17]:


ad_data=ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
correlation_matrix=ad_data.corr()
sns.heatmap(correlation_matrix, annot=True)


# In[18]:


ad_data=ad_data[['Area Income','Daily Internet Usage']]
correlation_matrix=ad_data.corr()
sns.heatmap(correlation_matrix, annot=True)


# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[16]:


ad_data=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[17]:


ad_data


# In[18]:


selected_columns = ['Country','City']

# Original DataFrame with categorical variables
print("Original DataFrame:")
print(ad_data[selected_columns].head())

label_encoder = LabelEncoder()
for col in selected_columns:
    ad_data[col + '_label'] = label_encoder.fit_transform(ad_data[col])
    
    
    # Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(ad_data[[col + '_label' for col in selected_columns]].head())


# In[17]:


label_encoder = LabelEncoder()
for col in selected_columns:
    ad_data[col + '_label'] = label_encoder.fit_transform(ad_data[col])


# In[18]:


# Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(ad_data[[col + '_label' for col in selected_columns]].head())


# In[20]:


one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(ad_data[selected_columns])


# In[25]:


column_names = one_hot_encoder.get_feature_names_out(selected_columns)
df= pd.DataFrame(one_hot_encoded, columns=column_names)
print("\nDataFrame after One-Hot Encoding:")
print(df.head())


# In[22]:


ad_data


# In[27]:


df


# In[28]:


numeric_columns=ad_data.select_dtypes(include='number')
correlation_matrix=numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True)


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)


# In[3]:


df


# In[6]:


df


# In[7]:


sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')


# In[8]:


sns.boxplot(x='sepal_length',y='sepal_width',data=df)


# In[9]:


sns.boxplot(data=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])


# In[10]:


from scipy.stats import zscore

z_scores = zscore(df['sepal_width'])
outlier_indices = (z_scores > 3) | (z_scores < -3)


# In[13]:


# Removing outliers
df_no_outliers = df[~outlier_indices]

# Compare the original and cleaned dataframes
print("Original DataFrame Shape:", df.shape)
print("DataFrame with Outliers Removed Shape:", df_no_outliers.shape)


# In[ ]:




