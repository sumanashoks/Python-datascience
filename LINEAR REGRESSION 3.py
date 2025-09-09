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



# In[3]:


df=pd.read_csv('/Users/apple/Desktop/python/Linear Regression 3/Linear Reg - Sale Forecast/Ecommerece.csv')


# In[4]:


df


# In[6]:


df.isnull().sum()


# In[13]:


df.drop(['Email','Avatar'],inplace=True, axis=1)


# In[14]:


df


# In[15]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[16]:


sns.pairplot(data=df)


# In[17]:


sns.boxplot(data=df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']])
plt.show()


# In[18]:


sns.distplot(df['Yearly Amount Spent'])


# In[22]:


reg=linear_model.LinearRegression()
#reg.fit(df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']],df.Yearly Amount Spent)
reg.fit(df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']], df['Yearly Amount Spent'])


# In[23]:


df.head()


# In[24]:


reg.predict([[34,12,38,8]])


# In[25]:


from sklearn.metrics import r2_score, mean_squared_error

# Assuming df is your DataFrame containing the data

# Make predictions
predicted_Prices = reg.predict(df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']])

# Calculate R-squared
r_squared = r2_score(df['Yearly Amount Spent'], predicted_Prices)
print(f"R-squared: {r_squared}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['Yearly Amount Spent'], predicted_Prices)
print(f"Mean Squared Error (MSE): {mse}")


# In[27]:


# Assuming df is your DataFrame containing the data
# Assuming reg is your trained linear regression model

# Make predictions on the entire dataset
predicted_prices = reg.predict(df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']])

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Yearly Amount Spent'], predicted_prices, alpha=0.5)

# Label axes and title
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Show plot
plt.show()


# # NOW I'll USE TRAIN TEST SPILT

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# In[2]:


df=pd.read_csv('/Users/apple/Desktop/python/Linear Regression 3/Linear Reg - Sale Forecast/Ecommerece.csv')


# In[3]:


df.drop(['Email','Address','Avatar'],inplace=True, axis=1)


# In[4]:


df


# In[5]:


X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[6]:


reg=linear_model.LinearRegression()

# Train the model on the training data
reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(X_test)


# ### THE BELOW CODE IS USED FOR WITHOUT USING TRAIN TEST SPILT.

# In[9]:


from sklearn.metrics import r2_score, mean_squared_error

# Assuming df is your DataFrame containing the data

# Make predictions
predicted_Prices = reg.predict(df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']])

# Calculate R-squared
r_squared = r2_score(df['Yearly Amount Spent'], predicted_Prices)
print(f"R-squared: {r_squared}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['Yearly Amount Spent'], predicted_Prices)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error (MSE)
rmse = mse ** 0.5
print("Root Mean Squared Error:", rmse)


# ### THE BELOW CODE REPRESENTS THE USING TRAIN TEST SPILT.
# 

# In[8]:


mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[41]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()


# In[10]:


reg.predict([[34,12,38,8]])


# In[11]:


reg.score(X_test,y_test)


# In[12]:


reg.score(X_train,y_train)


# In[ ]:




