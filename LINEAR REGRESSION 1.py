#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/ML/1_linear_reg/homeprices.csv')


# In[3]:


data


# In[4]:


plt.scatter(data.area,data.price,color='red',marker='*')#sns.scatterplot(x='area',y='price',data=data)
plt.xlabel('area')
plt.ylabel('price')


# In[5]:


sns.scatterplot(x='area',y='price',data=data) #plt.scatter(data.area,data.price,color='red',marker='*')


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


data.groupby('price').value_counts()


# In[10]:


reg=linear_model.LinearRegression()
reg.fit(data[['area']],data.price)


# # prdicting using area sqft how much will be the price

# In[11]:


reg.predict([[1200]])


# In[12]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming you've already fitted your model 'reg' and made predictions
predicted_values = reg.predict(data[['area']])

# Calculating R-squared
r_squared = r2_score(data.price, predicted_values)
print("R-squared:", r_squared)

# Calculating Mean Squared Error
mse = mean_squared_error(data.price, predicted_values)
print("Mean Squared Error:", mse)   #MSE LESSER THE VALUE BETTER THE RESULT

# Calculating Mean Absolute Error
mae = mean_absolute_error(data.price, predicted_values)
print("Mean Absolute Error:", mae)


# In[13]:


reg.coef_


# In[14]:


reg.intercept_


# # FORMULA
# 
# # Y=M X + b

# # we considered 
# 
# # X as area
# # y as price

# In[ ]:





# # now i want to predict  a set of data so i'll import new file 

# In[15]:


data1=pd.read_csv('/Users/apple/Desktop/python/ML/1_linear_reg/areas.csv')
data1.head(3)


# In[16]:


reg.predict(data1[['area']])


# In[17]:


p=reg.predict(data1[['area']])


# In[18]:


data1['prices_predicted']=p


# In[19]:


data1


# # prices get inflated

# In[20]:


data1['infaltion'] = data1['prices_predicted'] * 1.06


# In[21]:


data1


# # saving model
# 

# In[22]:


import pickle


# In[23]:


pickle.dump(reg,open('/Users/apple/Desktop/python/save/linear reg','wb'))


# In[24]:


modelloaded=pickle.load(open('/Users/apple/Desktop/python/save/linear reg','rb'))


# In[25]:


modelloaded.predict([[75000]])


# In[26]:


sa.predict([[70000]])


# #  SAVING MODEL USING JOBLIB

# In[ ]:


import joblib


# In[ ]:


joblib.dump(reg,'line reg joblib')


# In[ ]:


mj=joblib.load('line reg joblib')


# In[ ]:


mj.predict([[70000]])


# # _________________________________________________________________________

# # LINEAR REGRESSION WITH MULTIVERSE

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# In[27]:


df=pd.read_csv('/Users/apple/Desktop/python/ML/2_linear_reg_multivariate/homeprices.csv')


# In[28]:


df


# In[29]:


df.isnull().sum()


# In[30]:


df['bedrooms'].fillna(df['bedrooms'].median(),inplace=True)


# In[31]:


df


# In[45]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[47]:


sns.pairplot(data=df)


# In[32]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


# # Predicting the price using area bedroom and age

# In[33]:


reg.predict([[1200,3.0,20]])


# In[48]:


from sklearn.metrics import r2_score, mean_squared_error

# Assuming df is your DataFrame containing the data

# Make predictions
predicted_prices = reg.predict(df[['area','bedrooms','age']])

# Calculate R-squared
r_squared = r2_score(df['price'], predicted_prices)
print(f"R-squared: {r_squared}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(df['price'], predicted_prices)
print(f"Mean Squared Error (MSE): {mse}")


# # USING TRAIN TEST FOR THE SAME DATA
# 

# In[69]:


X = df[['area', 'bedrooms', 'age']]  # Features
y = df['price'] 


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[71]:


reg=linear_model.LinearRegression()
reg.fit(X_train, y_train)


# In[72]:


X_test


# In[73]:


predictions = reg.predict(X_test)


# In[74]:


reg.score(X_test,y_test)


# In[75]:


y_pred=reg.predict(X_test)


# In[77]:


reg.predict([[1200,3.0,20]])


# In[76]:


# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[78]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)


# In[81]:


# Assuming df is your DataFrame containing the data
# Assuming reg is your trained linear regression model

# Make predictions on the entire dataset
predicted_prices = reg.predict(df[['area', 'bedrooms', 'age']])

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['price'], predicted_prices, alpha=0.5)

# Label axes and title
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Show plot
plt.show()


# In[80]:


# Visualize predicted vs. actual values
plt.scatter(y_test, y_pred)

plt.show()


# ### AFTER USING TRAIN FOR LINEAR MY ACCURACY GOT REDUCED FROM 95 TO 92
# ### MY PRICE PREDICATION HAS REDUCED 36LAKHS TO 35 LAKHS 

# # ______________________________________________________________________

# # Let's practise an excercise problem

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model



# In[35]:


data=pd.read_csv('/Users/apple/Desktop/python/ML/2_linear_reg_multivariate/Exercise/hiring.csv')


# In[36]:


data


# ## I found null vales in Two columns
# 
# ## first column i made 0 due to experience
# 
# ## but second column i can't make '0' it's a score so i used median

# In[37]:


data['experience'].fillna(0,inplace=True)


# In[38]:


data


# ## I installed word2number
# ## because first column has numbers but its in characters so only i used this

# In[39]:


pip install word2number


# ## after installing we need to convert the column to number

# In[40]:


from word2number import w2n
# Assuming 'experience' is the column that needs conversion
data['experience'] = data['experience'].apply(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x)



# In[41]:


data


# ## I used median to column 2 since it's useful data insight

# In[42]:


data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].median(),inplace=True)


# In[43]:


data


# In[44]:


reg=linear_model.LinearRegression()
reg.fit(data[['experience','test_score(out of 10)','interview_score(out of 10']],data.salary($))


# ## u can see the above code i've used corrrectly but it throwed me an error
# ## because the salary column has (dollar_symbol) so I've to insert a new square barcket
# ## in nxt line of code it got executed

# In[ ]:


reg=linear_model.LinearRegression()
reg.fit(data[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], data['salary($)'])


# ## I predicted SALARY
# ## 2yrs experience, 9 test score, 6 interview score

# In[ ]:


reg.predict([[2,9,6]])


# ## I predicted SALARY
# ## 12yrs experience, 10 test score, 10 interview score

# In[ ]:


reg.predict([[12,10,10]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




