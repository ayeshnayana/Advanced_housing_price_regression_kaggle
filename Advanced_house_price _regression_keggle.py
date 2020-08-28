#!/usr/bin/env python
# coding: utf-8
## Data set https://www.kaggle.com/c/house-prices-advanced-regression-techniques
## Create the anaconda environment houseprice conda create -n houseprice python=3.7
## Install all the required packages using pip install <package name>
## Import all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
## We want to see all the columns 
pd.pandas.set_option('display.max_columns',None)
# In[4]:


## Reading the training data set
dataset=pd.read_csv('train.csv')
## Print shape of dataset with 


# In[5]:


dataset.head()


# In[6]:


print(dataset.shape)


# In[10]:


## We are now looking for the null values in the data set
## Loop through the different fetures in the data set and put an if condition, which filters the fetures that have summation of null values greater than 1
## features_with_na is an array contains all the fetures that have null values 
features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>>1]
## Now we gonna print the fetures that have null values and the precentage of null values per each feature
## Loop through different fetures with null values and we then round the mean of the null values upto 4 decimal places
for feature in features_with_na:
    print(feature,np.round(dataset[feature].isnull().mean(),4),'% missing values')


# # Missing values and their relationship to sales price

# In[13]:


## We gonna plot some features.
## Want to know how does the features with missing values affect the data.
## If there is no much effect from these features we can drop them 
## If there is a high effect then we have to handle the missing values 
for feature in features_with_na:
    #first make a copy of the data set
    data=dataset.copy()
    
    ## Lets make a varible which give 1 everytime it encounters missing value.
    ## Else get zero 
    data[feature]=np.where(data[feature].isnull(),1,0)
    ## From this we gonna plot a graph of sales price vs the missing values.
    ## This graph can show us whether having lot of missing values couses a rise in sales price
    ## Or it might show us whether having lot of missing values can decrease the sales price
    ## Lets calculate the mean sales price for the each feature that have missing fetures
    data.groupby(feature)['SalePrice'].median().plot.bar()
    ## Now we group the salesprice with features. i.e we have 1 for each missing value then we plot the salesprice for that missing value
    ## We also have 0 for each available data and we graouped them together so that we can find the mean sales price for these data and plot
    plt.title(feature)
    plt.show()
    

From these graphs we infer that when there are missing values it will affect the sale price of the house. This means we need to use fature engineering to figure out what values we must use to replace these missing values.
# In[15]:


## Also, we find that the id of houses is redundant. Need to drop it from the dataset
print("ID of houses {}".format(len(dataset.Id)))


# In[17]:


## Now we need to find the features that contain only numerical values 
## Here we loop through all the features that does not include data structure "O"
numerical_features=[features for features in dataset.columns if dataset[features].dtypes !='O']

print('Number of numerical variables: ',len(numerical_features))

dataset[numerical_features].head()


# # Temporal Variables
There are 4 temporal variables in the data set. Go through the numerical data and we see YearBuilt, YearRemodAdd, GarageYrBlt and YrSold has temporal data. We need to extract information such as number of years from these data.
# In[18]:


## Extracting the number of years from the temporal fetures
## We will loop through all the temporal features to see whether there are only 4 temporal features.
## In the loop we are looking for a key word "Yr"
year_feature= [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# In[19]:


## Explore the content in the temporal features
for feature in year_feature:
    print(feature,dataset[feature].unique())

Now we are going to analyze the temporal features.
First we will check whether there is a correlation beween these temporal features and the sale price by plotting them.
We will group the data by temporal feature and then plot the sale price against it.
# In[28]:


dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('YearSold')
plt.ylabel('Median house price')  
plt.title('House price vs year sold')

The sale price go down as the year sold increases. This does not happen in the real world unless there are other factors that does not play a role in pricing.
# In[29]:


dataset.groupby('YearBuilt')['SalePrice'].median().plot()
plt.xlabel('YearBuilt')
plt.ylabel('Median house price')  
plt.title('House price vs Year Built')

The house price increases when the house is new. This is what we expect from the real world experience. There is a spike of at the 1880-1900. This is because when the house has a historical importance it gets a higher selling price.
# In[30]:


dataset.groupby('YearRemodAdd')['SalePrice'].median().plot()
plt.xlabel('YearRemodAdd')
plt.ylabel('Median house price')  
plt.title('House price vs Year Remod Add')


# In[31]:


dataset.groupby('GarageYrBlt')['SalePrice'].median().plot()
plt.xlabel('GarageYrBlt')
plt.ylabel('Median house price')  
plt.title('House price vs Year GarageYrBlt')


# In[33]:


## Since all other features has a distribution we expect from real world, we would like to analyze why the year sold was different.
## We will plot the sale price vs YrSold-other_features
## This graph will show us the variation of the sale price without the effect of the additional feature.
for feature in year_feature:
    if feature != 'YrSold':
        ## first make a copy of dataset
        data=dataset.copy()
        ## Now we find the difference between the YrSold and the each feature
        ## Make a new feature called feature from this
        data[feature]=data['YrSold']-data[feature]
        
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('sale price')
        plt.show()


# From the above figures we infer that when the house is new or it is remodeled recenly and added a garad recently the sale price go up. The difference tells us the number of years has passed between the selling year and the feature added year.
There are two types of numerical variables. 
    1. Continuouse variables
    2. Descrete variables 
# # Descrete variables 

# In[35]:


## To figure out what are the descrete features we will loop through the data set
## We look for the feature that have less than 25 different features. 
## The Id feature and year features can have more less than 25 different values so we exclude them.
descrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id'] ]
print("Descrete Variable Count: {}".format(len(descrete_feature)))


# In[36]:


descrete_feature


# In[37]:


dataset[descrete_feature].head()

Now we gonna analyze the effect of the descrete feature to the sale price. For this we plot all these features against the sale price.
# In[41]:


## To plot the descrete feature against the sales price
for feature in descrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Median sale price')
    plt.show()

Observation:
    We find that the sale price exponentially increase as the over quality of house and the number of full baths increases. 
    The sale price increase as the number of cars that can be parked (garage size) increase up to 3 cars.
    Having more than 2 fire places increases the price.
    When having lots of rooms above the ground level increases the price as well.
    Month sold does not matter much
# # Continuouse variables 

# In[43]:


##Look for the continuouse features
continuouse_features=[feature for feature in numerical_features if feature not in descrete_feature+['Id']]
print("Number of continuouse features: {}".format(len(continuouse_features)))


# In[44]:


continuouse_features


# In[48]:


for feature in continuouse_features:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()


# In[ ]:




