#!/usr/bin/env python
# coding: utf-8

# # LINEAR  REGRESSION: Tesla Stock Price Prediction

# In[25]:


#Linear regression is the relationship between dependent and independent variable(feature/predictive variable)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
tesla_file_path= (r'C:\Users\ansht\Downloads\TSLA.csv')
tesla_df= pd.read_csv(tesla_file_path)


# In[26]:


tesla_df.head(5)
#displaying the first 5 columns


# In[27]:


tesla_df.info()


# In[28]:


tesla_df.describe()
#to get statistical information regarding the dataset


# # LIST

# In[29]:


list(tesla_df.columns)
#Creating a list of the columns being used


# In[30]:


#to check if any null values are there or not in the dataset
tesla_df.isnull()
#gives false means no null values in the dataset and so their sum is also 0


# In[31]:


#to calculate the sum of null values. As sum is 0 it will return all the values as 0
tesla_df.isnull().sum()


# In[32]:


#to display the null values dropna() method is used
tesla_df.dropna(axis=0,how='any', thresh=None, subset=None,inplace=False)
#as no null values are obtained so will display all


# In[33]:


#as the features like Adj Close and Close are same dropping them would be a great idea and Date is also not useful Categorial Data


# In[34]:


tesla_df


# In[35]:


from sklearn.linear_model import LinearRegression
#importing it from a library called sklearn
teslamodel= LinearRegression() 


# # FEATURE SELECTION

# In[36]:


#Step 2 Selecting a prediction target and Separating target and Features
tesla_features=['Open','High','Volume','Low']
X=tesla_df[tesla_features]


# In[37]:


#Creating an x array as we'll use it for the prediction of y
x=tesla_df[['Open','High','Volume','Low']]


# In[38]:


#Creating an array y for the value of which we want to predict
y=tesla_df['Close']


# # SPLITING DATA

# In[39]:


#from sklearn import the libraries
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#training data into 70% and testing as 30%


# # BUILDING MODEL

# In[40]:


from sklearn.linear_model import LinearRegression
#creating a linear regressor model
model=LinearRegression()


# # FITTING THE MODEL

# In[41]:


#as a overfit model doesn't perform well on the training data and rather on the validation part
tesla_df=model.fit(x_train,y_train)


# In[42]:


print(model.coef_)
#printing model's coefficients and intercepts


# In[43]:


print(model.intercept_)


# # PREDICTING THE MODEL

# In[44]:


#predict only takes parameter x and gives automatically the values of y. Compare y with the predictions parameter to get the accuracy
prediction=model.predict(x_test)


# In[45]:


comparison = pd.DataFrame({'Predicted Values':prediction,'Actual Values':y_test})


# In[46]:


print (comparison.head(10))


# # ACCURACY OF MODEL: MEAN ABSOLUTE ERROR

# # CROSS VALIDATION

# In[47]:


#mean absolute error is actually actual value-predicted value divided by n number of observations
from sklearn import metrics 
metrics.mean_absolute_error(y_test, prediction)


# # ROOT MEAN SQAURE VALUE(RMSE)

# In[ ]:


#Note:RMSE will always be greater or equal to the MAE value. MSE(RMSE=sqrt(Mean Square Error))
#No inbuilt method to find the RMSE in sciikit model
metrics.mean_squared_error(y_test, prediction)


# In[ ]:


#Calculating RMSE
np.sqrt(metrics.mean_squared_error(y_test, prediction))


# # DATA VISUALIZATION

# # SCATTER PLOT

# In[ ]:


#Scatterplot shows relation between two continous variables so x and y
plt.scatter(y_test,prediction,color='red')
#A perfectly straight diagonal line in this scatterplot would indicate that our model perfectly predicted the y-array values.


# In[ ]:


sns.set_style("dark")


# # HISTOGRAM

# In[ ]:


plt.hist(y_test-prediction)


# # BARPLOT

# In[ ]:


plt.bar(y_test,prediction)


# In[ ]:


sns.regplot(y_test,prediction)

