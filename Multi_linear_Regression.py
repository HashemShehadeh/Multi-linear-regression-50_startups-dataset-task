# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:51:12 2022

@author: hashem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

#Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct= ColumnTransformer(transformers=[("ecoder",OneHotEncoder(),[3])], remainder="passthrough")
X=np.array(ct.fit_transform(X))

#to reduse the dimentionality
X=X[:,1:]


#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size= 0.2 , random_state=0)

'''
#feature Scaling
#On the linear regression no need to feature scaling
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train=std.fit_transform(X_train)
X_test=std.fit_transform(X_test)
'''
#start the multi linear regression

from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(X_train,y_train)

y_pred= lr.predict(X_test)


#to treat with the constatnt value
X=np.insert(X,0,1,axis=1)
#Or
#X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

#applying the evaluation method (Backward Elimination)
import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=np.array(X[:,[0,1,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=np.array(X[:,[0,3,4,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=np.array(X[:,[0,3,5]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()


from sklearn.model_selection import train_test_split

X_train_opt,X_test_opt,y_train,y_test= train_test_split(X_opt,y, test_size= 0.2 , random_state=0)

from sklearn.linear_model import LinearRegression

lr2= LinearRegression()
lr2.fit(X_train_opt,y_train)

y_pred2= lr2.predict(X_test_opt)