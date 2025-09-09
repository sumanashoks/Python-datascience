#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


df=pd.read_csv('/Users/apple/Downloads/car_evaluation.csv')


# In[3]:


df


# In[4]:


from sklearn.preprocessing import LabelEncoder

selected_columns = ['vhigh','vhigh.1','2','2.1','small','low','unacc'] #vhigh	vhigh.1	2	2.1	small	low	unacc

# Original DataFrame with categorical variables
print("Original DataFrame:")
print(df[selected_columns].head())

label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])
    
    
    # Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[5]:


label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])


# In[6]:


# Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[12]:


#df.drop(['vhigh','vhigh.1','2','2.1','small','low','unacc']axis=1,inplace=True)
df.drop(['vhigh', 'vhigh.1', '2', '2.1', 'small', 'low', 'unacc'], axis=1, inplace=True)


# In[13]:


df


# In[17]:


sns.countplot(x='unacc_label',data=df)


# In[18]:


sns.countplot(x='low_label',data=df)


# In[19]:


sns.countplot(x='small_label',data=df)


# In[20]:


sns.countplot(x='vhigh.1_label',data=df)


# In[22]:


sns.countplot(x='vhigh_label',data=df)


# In[14]:


df.info()


# In[15]:


df['unacc_sa'].unique()


# In[16]:


df.columns


# In[52]:


from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1]
y=df.iloc[:, -1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.28,random_state=41)


# In[53]:


X


# In[54]:


y


# In[55]:


reg=linear_model.LogisticRegression()
reg.fit(X_train,y_train)


# In[56]:


reg.predict(X_test)


# In[57]:


reg.score(X_test,y_test)


# In[58]:


reg.score(X_train,y_train)


# In[59]:


df.head(n=3)


# In[60]:


reg.predict([[3,3,1,1,2,1]])


# In[61]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming you've already fitted your model 'reg' and made predictions
predicted_values = reg.predict(df[['vhigh_sa','vhigh.1_sa','2_sa','2.1_sa','small_sa','low_sa']])

# Calculating R-squared
r_squared = r2_score(df.unacc_sa, predicted_values)
print("R-squared:", r_squared)

# Calculating Mean Squared Error
mse = mean_squared_error(df.unacc_sa, predicted_values)
print("Mean Squared Error:", mse)   #MSE LESSER THE VALUE BETTER THE RESULT

# Calculating Mean Absolute Error
mae = mean_absolute_error(df.unacc_sa, predicted_values)
print("Mean Absolute Error:", mae)


# In[ ]:





# #  now i'm using imbalancing method

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


df=pd.read_csv('/Users/apple/Downloads/car_evaluation.csv')


# In[3]:


df


# In[4]:


from sklearn.preprocessing import LabelEncoder

selected_columns = ['vhigh','vhigh.1','2','2.1','small','low','unacc'] #vhigh	vhigh.1	2	2.1	small	low	unacc

# Original DataFrame with categorical variables
print("Original DataFrame:")
print(df[selected_columns].head())

label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])
    
    
    # Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[5]:


label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])


# In[6]:


# Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[7]:


#df.drop(['vhigh','vhigh.1','2','2.1','small','low','unacc']axis=1,inplace=True)
df.drop(['vhigh', 'vhigh.1', '2', '2.1', 'small', 'low', 'unacc'], axis=1, inplace=True)


# In[8]:


df


# In[24]:


sns.countplot(x='unacc_sa',data=df)


# In[11]:


from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1]
y=df.unacc_sa
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=59)
X.head()


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[14]:


from imblearn.over_sampling import SMOTE
smote=SMOTE()


# In[15]:


X_train_smote,y_train_smote=smote.fit_resample(X_train.astype('float'),y_train)


# In[16]:


from collections import Counter

print('Before SMOTE:', Counter(y_train))
print('After SMOTE:', Counter(y_train_smote))


# In[17]:


model.fit(X_train_smote,y_train_smote)
y_predict=model.predict(X_test)

print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_train_smote is the target variable after SMOTE
sns.countplot(x=y_train_smote)
plt.title('Count Plot after SMOTE')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()


# In[18]:


reg=linear_model.LogisticRegression()
reg.fit(X_train_smote,y_train_smote)


# In[19]:


reg.score(X_train_smote,y_train_smote)


# In[20]:


reg.score(X_test,y_test)


# In[21]:


df.head(5)


# In[22]:


reg.predict([[3,3,1,1,1,2]])


# In[23]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming you've already fitted your model 'reg' and made predictions
predicted_values = reg.predict(df[['vhigh_sa','vhigh.1_sa','2_sa','2.1_sa','small_sa','low_sa']])

# Calculating R-squared
r_squared = r2_score(df.unacc_sa, predicted_values)
print("R-squared:", r_squared)

# Calculating Mean Squared Error
mse = mean_squared_error(df.unacc_sa, predicted_values)
print("Mean Squared Error:", mse)   #MSE LESSER THE VALUE BETTER THE RESULT

# Calculating Mean Absolute Error
mae = mean_absolute_error(df.unacc_sa, predicted_values)
print("Mean Absolute Error:", mae)


# # Another imbalance method
# # 
# # use this method
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


# In[2]:


df=pd.read_csv('/Users/apple/Downloads/car_evaluation.csv')


# In[3]:


df


# In[4]:


from sklearn.preprocessing import LabelEncoder

selected_columns = ['vhigh','vhigh.1','2','2.1','small','low','unacc'] #vhigh	vhigh.1	2	2.1	small	low	unacc

# Original DataFrame with categorical variables
print("Original DataFrame:")
print(df[selected_columns].head())

label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])
    
    
    # Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[5]:


label_encoder = LabelEncoder()
for col in selected_columns:
    df[col + '_sa'] = label_encoder.fit_transform(df[col])


# In[6]:


# Transformed DataFrame using Label Encoding
print("\nDataFrame after Label Encoding:")
print(df[[col + '_sa' for col in selected_columns]].head())


# In[7]:


#df.drop(['vhigh','vhigh.1','2','2.1','small','low','unacc']axis=1,inplace=True)
df.drop(['vhigh', 'vhigh.1', '2', '2.1', 'small', 'low', 'unacc'], axis=1, inplace=True)


# In[8]:


df


# In[9]:


sns.countplot(x='unacc_sa',data=df)


# In[10]:


# Read the document end for detailed notes on imbalance data
# Handling class imbalance with SMOTE
X = df.drop('unacc_sa', axis=1)
y = df['unacc_sa']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[11]:


sns.countplot(x=y_resampled)
plt.show()


# In[12]:


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.40, random_state=101)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit( X_train,y_train)

# Make predictions
y_pred = model.predict(X_test)


# In[13]:


from collections import Counter

print('Before SMOTE:', Counter(y_train))
print('After SMOTE:', Counter(y_resampled))


# In[14]:


accuracy = accuracy_score(y_pred,y_test)
print("Accuracy:", accuracy)


# In[15]:


model.score(X_test,y_test)


# In[16]:


model.score(X_train,y_train)


# In[17]:


model.predict([[3,3,1,1,2,1]])


# In[18]:


reg=linear_model.LogisticRegression()
reg.fit(X_train,y_train)


# In[19]:


reg.predict(X_test)


# In[20]:


reg.score(X_train,y_train)


# In[21]:


reg.score(X_test,y_test)


# In[22]:


reg.predict([[3,3,1,1,2,1]])


# In[23]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming you've already fitted your model 'reg' and made predictions
predicted_values = reg.predict(df[['vhigh_sa','vhigh.1_sa','2_sa','2.1_sa','small_sa','low_sa']])

# Calculating R-squared
r_squared = r2_score(df.unacc_sa, predicted_values)
print("R-squared:", r_squared)

# Calculating Mean Squared Error
mse = mean_squared_error(df.unacc_sa, predicted_values)
print("Mean Squared Error:", mse)   #MSE LESSER THE VALUE BETTER THE RESULT

# Calculating Mean Absolute Error
mae = mean_absolute_error(df.unacc_sa, predicted_values)
print("Mean Absolute Error:", mae)


# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you've already fitted your model 'reg' and made predictions on the test set 'X_test'
predictions = reg.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)


# ### I Got Confused Why Both Train and Test Score are same 56
# 
# ## but belove code will resolve the confusion

# In[25]:


from sklearn.model_selection import train_test_split

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Training the model
reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)

# Checking the scores again
train_score = reg.score(X_train, y_train)
test_score = reg.score(X_test, y_test)

print("Training Score:", train_score)
print("Testing Score:", test_score)


# In[ ]:




