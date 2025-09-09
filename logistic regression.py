#!/usr/bin/env python
# coding: utf-8

# ## usually logistics regression is a classification type like 0/1, True/False, Yes/No
# ## so when classification comes we train the data why because to check the accuracy
# ## how well the data is good for future testing and predicting

# ## if the accuracy is good the predicting number will good
# ## if the accuracy is bad will get wrong predicted numbers

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/ML/7_logistic_reg/insurance_data.csv')


# In[3]:


data


# In[4]:


sns.boxplot(data=data,x='age')


# In[5]:


sns.boxplot(data=data,x='bought_insurance')


# # I found outliers in age column try to remove it

# In[6]:


sns.countplot(data=data,x='bought_insurance')


# In[7]:


data.bought_insurance.value_counts()


# # HANDLING IMBALANCE DATA

# In[8]:


from sklearn.model_selection import train_test_split
X=data.iloc[:,:-1]
y=data.bought_insurance
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=59)
X.head()


# In[9]:


y.head()


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[12]:


from imblearn.over_sampling import SMOTE
smote=SMOTE()


# In[13]:


X_train_smote,y_train_smote=smote.fit_resample(X_train.astype('float'),y_train)


# In[14]:


from collections import Counter

print('Before SMOTE:', Counter(y_train))
print('After SMOTE:', Counter(y_train_smote))


# In[15]:


model.fit(X_train_smote,y_train_smote)
y_predict=model.predict(X_test)

print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[16]:


reg=linear_model.LogisticRegression()
reg.fit(X_train_smote,y_train_smote)


# In[17]:


reg.score(X_train_smote,y_train_smote)


# In[18]:


reg.score(X_test,y_test)


# In[ ]:





# # ANOTHER METHOD OF HANDLING IMBALNCE DATA

# In[25]:


# Read the document end for detailed notes on imbalance data
# Handling class imbalance with SMOTE
X = data.drop('bought_insurance', axis=1)
y = data['bought_insurance']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[60]:


sns.countplot(x=y_resampled)


# In[62]:


from sklearn.linear_model import LogisticRegression

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.35, random_state=101)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[50]:


# Check the column names in X_resampled
print(X_resampled.columns)


# ### after sampling it got reduced from 85 to 80

# # _____________________________________________________________
# # _____________________________________________________________

# In[ ]:





# # Another method using normal test and train 

# In[47]:


plt.scatter(data.age,data.bought_insurance,marker='*',color='red')
plt.plot([10,60], [0,1], color='blue')
plt.grid()


# In[28]:


plt.scatter(data.age,data.bought_insurance,marker='*',color='red')
plt.plot([10,20,30, 45,50,55,60], [0,0,0,1,1,1,1], color='blue')
plt.grid()


# ## The above graph represents
# ## people with age between 10 to 30 havn't bought insurance.  and
# ## people with age between greater than 40 bought insurance 

# # Now will train the model

# In[36]:


from sklearn.model_selection import train_test_split
X=data[['age']]
y=data[['bought_insurance']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=59)


# In[37]:


X_test


# In[38]:


reg=linear_model.LogisticRegression()
reg.fit(X_train,y_train)


# In[39]:


reg.predict(X_test)


# # Now I'll check the accuracy

# In[40]:


reg.score(X_test,y_test)


# In[45]:


reg.score(X_train,y_train)


# # now I'll predict the some random age which isn't there in data
# # age 29

# In[42]:


reg.predict([[29]])


# # It gave me '0'
# # '0' states that insurance never bought
# # '1' states that insurance was bought

#     ||
#     ||
#     ||
#   \ || /
#    \  /
#     \/
#     
#     
# 
# 
# 
# 
# 
# # checking accuracy for predicted values

# In[43]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming you've already fitted your model 'reg' and made predictions
predicted_values = reg.predict(data[['age']])

# Calculating R-squared
r_squared = r2_score(data.bought_insurance, predicted_values)
print("R-squared:", r_squared)

# Calculating Mean Squared Error
mse = mean_squared_error(data.bought_insurance, predicted_values)
print("Mean Squared Error:", mse)   #MSE LESSER THE VALUE BETTER THE RESULT

# Calculating Mean Absolute Error
mae = mean_absolute_error(data.bought_insurance, predicted_values)
print("Mean Absolute Error:", mae)


# In[35]:


plt.figure(figsize=(8, 6))

# Scatter plot for actual data
plt.scatter(data['age'], data['bought_insurance'], color='blue', label='Actual')

# Scatter plot for predicted values
plt.scatter(data['age'], predicted_values, color='red', label='Predicted')

# Plotting the regression line
plt.plot(data['age'], reg.predict(data[['age']]), color='green', label='Regression Line')

plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


# # SAVING THE MODEL USING PICKLE

# In[18]:


import pickle


# In[19]:


pickle.dump(reg,open('/Users/apple/Desktop/python/save/logistic reg','wb'))


# In[20]:


modelloaded=pickle.load(open('/Users/apple/Desktop/python/save/logistic reg','rb'))


# In[21]:


modelloaded.predict([[23]])


# In[ ]:





# # Tried ploynomial feature to get better accuracy for predi

# In[22]:


from sklearn.preprocessing import PolynomialFeatures

# Assuming X_train, y_train, X_test, y_test are already defined

# Applying Polynomial Regression with degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

reg_poly = linear_model.LinearRegression()
reg_poly.fit(X_train_poly, y_train)

# Making predictions with the polynomial model
predicted_values_poly = reg_poly.predict(X_test_poly)

# Calculating R-squared for polynomial regression
r_squared_poly = r2_score(y_test, predicted_values_poly)
print("R-squared (Polynomial Regression):", r_squared_poly)


# In[23]:


plt.figure(figsize=(8, 6))

# Scatter plot for actual test data
plt.scatter(X_test, y_test, color='blue', label='Actual')

# Scatter plot for predicted values using polynomial regression
plt.scatter(X_test, predicted_values_poly, color='red', label='Predicted (Polynomial Regression)')

# Plotting the regression line for polynomial regression
plt.scatter(X_test, reg_poly.predict(X_test_poly), color='green', label='Regression Line (Polynomial)')

plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Actual vs Predicted Values (Polynomial Regression)')
plt.legend()
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))

# Scatter plot for original data
plt.scatter(X_test, y_test, color='blue', label='Actual')

# Scatter plot for predictions using linear regression
plt.scatter(X_test, reg.predict(X_test), color='red', label='Linear Regression')

# Scatter plot for predictions using polynomial regression
plt.scatter(X_test, reg_poly.predict(X_test_poly), color='green', label='Polynomial Regression')

plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Comparison of Predictions')
plt.legend()
plt.show()


# In[ ]:





# # L1 &L2
# 
# 

# In[25]:


from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train,y_train)


# In[26]:


from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train,y_train)


# In[27]:


ridge_reg.score(X_test,y_test)


# In[28]:


ridge_reg.score(X_train,y_train)


# In[ ]:





# In[ ]:





# # ________________________________________________________________________

# # _______________________________________________________________________

# # SAME PROBLEM
# # CHECKING ACCURACY USING decision tree
# 

# In[29]:


x=data[['age']]
y=data['bought_insurance']

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model=DecisionTreeClassifier()
model.fit(x,y)


# In[30]:


model.predict([[29]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




