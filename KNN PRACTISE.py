#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


ad_data=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[3]:


ad_data.head(n=10)


# In[4]:


ad_data.tail()


# In[5]:


ad_data.info()


# In[6]:


ad_data.describe()


# In[7]:


sns.histplot(ad_data['Daily Time Spent on Site'],kde=True)


# In[8]:


sns.jointplot(data=ad_data,x='Age',y='Daily Internet Usage')
plt.grid()
plt.show()


# In[9]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# In[10]:


ad_data.describe()


# # NOW START ALGORITHIM
# 

# # STEP 1
#    ## use scalling to eliminate higher number
#  
# # STEP 2
#    ## use train_test_split
#    ## to tarin the model
#  
# # STEP 3
#   ## use KNN --> when there is classification method always use KNN
#   ## and we use ELBOW method to find K
#   
# # STEP 4
#   ## keeping elbow method data we find accuracy by using confusion matrix,classification 
#   ## report
#   
#   # This step 4 we call model buliding and ( predict & evulate model)

# In[11]:


ad_data.describe()


# #  SCALLING

# In[12]:


from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
scaler.fit(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']])
scaler_features=scaler.transform(ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']])
df_feat=pd.DataFrame(scaler_features,columns=[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']])                                       


# # TRAIN_TEST_SPLIT

# In[13]:


# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X=df_feat 
y=ad_data['Clicked on Ad'] #why we give y=ad_data['Clicked on Ad'] because y is output so what u wanna find out you can give that coloumn in y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)


# # NOW KNN & ELBOW METHOD

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
error_rate=[]
for i in range (1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.plot(range(1,20),error_rate)


# # BUILDING MODEL AND PREDICT &EVULATING 
#  
#    ## we found 'K' IS "2" because the ELBOW is bent near 2 
#    ## we always consider first elbow bent "K"
#    ## we consider the dip i mean the lowest bottom line as neighbors
#    ## in this 12 is bottom & we consider as neighbors

# In[15]:


from sklearn.metrics import classification_report,confusion_matrix
#Build the Model with 2 as K
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ## ACCURACY 97%

# In[17]:


df_feat.to_csv('/Users/apple/Desktop/python/suman_clean_1.csv', index=False)


# # DECISION TREE

# ### This Decision tree is formed using all the columns. in such a case you can use this algorithim for decision

# In[17]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))


# In[18]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming you have already trained your decision tree model and named it 'dtree'

feature_names = list(X.columns.get_level_values(0))  # Extracting feature names from MultiIndex
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
plot_tree(dtree, filled=True, feature_names=feature_names)  # Plotting the decision tree
plt.title("Decision Tree Plot")
plt.show()


# # RANDOM FOREST

# In[19]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[21]:


rfc_pred=rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,pred))


# In[ ]:





# # Decision Tree

# ## This decision tree is used only for particular columns and you can optimize according to your criteria and use this algorithim

# In[38]:


ad_data.describe()


# In[59]:


x=ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y=ad_data['Clicked on Ad']


# In[60]:


from sklearn import tree
model=DecisionTreeClassifier()


# In[62]:


model.fit(x,y)


# In[63]:


model.score(x,y)


# In[64]:


ad_data


# ## the numbers i've inserted is randomly representing the numeric column of 'X' independent.
# ## and expecting the output of dependent column 'y 
# ## whether he /she clicked or not the ad

# In[65]:


model.predict([[80.23,35,61833.90,256.09,0]])


# In[66]:


model.predict([[60,20,60000,250,0]])


# In[68]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Assuming 'x' is your feature dataset and 'y' is your target variable
model = DecisionTreeClassifier()
model.fit(x, y)

plt.figure(figsize=(20, 10))  # Set the figure size for better visualization
plot_tree(model, filled=True, feature_names=x.columns.tolist())  # Plot the decision tree
plt.title("Decision Tree Visualization")
plt.show()


# # Random Forest with using Train test spilt

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[11]:


df=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[12]:


x=df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y=df['Clicked on Ad']


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=51)


# In[52]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[53]:


rf_model = RandomForestClassifier()

# Training the model on the scaled training data
rf_model.fit(x_train_scaled, y_train)

accuracy = rf_model.score(x_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Predict using the test set
y_pred = rf_model.predict(x_test_scaled)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# ## Random forest without using trainng test spilt

# In[55]:


df=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[56]:


x=df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y=df['Clicked on Ad']


# In[57]:


# Assuming 'x' is your feature dataset and 'y' is your target variable
model_rf = RandomForestClassifier()  # Create a Random Forest Classifier
model_rf.fit(x, y)  # Train the model

# Check model accuracy
accuracy = model_rf.score(x, y)
print("Random Forest Accuracy:", accuracy)

# Make predictions
new_data_point = [[80.23, 35, 61833.90, 256.09, 0]]
prediction = model_rf.predict(new_data_point)
print("Prediction for new data point:", prediction)

# Visualize a single tree within the Random Forest (first tree in this case)
plt.figure(figsize=(20, 10))  # Set the figure size for better visualization
tree.plot_tree(model_rf.estimators_[0], filled=True, feature_names=x.columns.tolist())  # Plot the first tree in the forest
plt.title("Random Forest Visualization - Single Tree")  # Set title
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




