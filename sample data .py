#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[15]:


df=pd.read_excel('/Users/apple/Documents/ONEHOT.XLSX')


# In[16]:


df


# In[17]:


# Select some columns with categorical variables
selected_columns = ['COMPANY', 'DEGREE', 'CITY']

# Original DataFrame with categorical variables
print("Original DataFrame:")
print(df[selected_columns].head())

# Label Encoding
label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_label'] = label_encoder.fit_transform(df[col])

# Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_label' for col in selected_columns]].head())

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[selected_columns])

# Transformed DataFrame using One-Hot Encoding
column_names = one_hot_encoder.get_feature_names_out(selected_columns)
df_one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=column_names)
print("\nDataFrame after One-Hot Encoding:")
print(df_one_hot_encoded.head())


# In[18]:


df


# In[19]:


df['COMPANY'].value_counts(normalize=True)*100


# In[20]:


df['CITY'].value_counts(normalize=True)*100


# In[21]:


df['DEGREE'].value_counts(normalize=True)*100


# In[22]:


Google=df[df['COMPANY']=='GOOGLE']
Google['DEGREE'].value_counts(normalize=True)*100


# In[ ]:





# In[ ]:




