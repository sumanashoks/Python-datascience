#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# In[2]:


df=pd.read_csv('/Users/apple/Desktop/python/Linear Regression 3/Linear Reg - Housing Price/USA_Housing.csv')


# In[3]:


df


# In[4]:


df.isnull()


# In[5]:


df.isnull().sum()


# In[6]:


df.drop(['Address'],axis=1,inplace=True)


# In[7]:


df


# In[8]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[9]:


sns.pairplot(df)


# In[10]:


sns.distplot(df['Price'])


# In[11]:


df.info()


# In[22]:


reg=linear_model.LinearRegression()
reg.fit(df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']],df.Price)


# In[23]:


df.describe()


# In[28]:


reg.predict([[107701.748378,9.519088,10.759588,6.500000,69621.713378]])


# In[29]:


from sklearn.metrics import r2_score, mean_squared_error

# Assuming df is your DataFrame containing the data

# Make predictions
predicted_Prices = reg.predict(df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']])

# Calculate R-squared
r_squared = r2_score(df['Price'], predicted_Prices)
print(f"R-squared: {r_squared}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['Price'], predicted_Prices)
print(f"Mean Squared Error (MSE): {mse}")


# In[30]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming df is your DataFrame containing the data
# Assuming 'Price' column represents the actual prices

# Make predictions
predicted_prices = reg.predict(df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']])

# Calculate R-squared
r_squared = r2_score(df['Price'], predicted_prices)
print(f"R-squared: {r_squared}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['Price'], predicted_prices)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(df['Price'], predicted_prices)
print(f"Mean Absolute Error (MAE): {mae}")


# In[50]:


# Assuming df is your DataFrame containing the data
# Assuming reg is your trained linear regression model

# Make predictions on the entire dataset
predicted_prices = reg.predict(df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']])

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Price'], predicted_prices, alpha=0.5)

# Label axes and title
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Show plot
plt.show()


# # now will use TRAIN TEST SPILT

# In[12]:


X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
  # Features
y = df['Price'] 


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[14]:


reg=linear_model.LinearRegression()
reg.fit(X_train, y_train)


# In[15]:


X_test


# In[16]:


predictions = reg.predict(X_test)


# In[23]:


reg.score(X_test,predictions)


# In[18]:


y_pred=reg.predict(X_test)


# In[19]:


reg.predict([[107701.748378,9.519088,10.759588,6.500000,69621.713378]])


# In[21]:


# Calculate metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[22]:


plt.scatter(y_test,predictions)


# In[ ]:




