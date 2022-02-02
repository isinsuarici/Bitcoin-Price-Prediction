# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:44:01 2021

@author: isinsu
"""
#1.libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor



# data
veriler = pd.read_csv('BTCUSDT.csv')

#veriler.tail()


#filling in missing data with the mean

imputer=SimpleImputer(missing_values=np.nan , strategy="mean")
eksikveriler= veriler.iloc[:,3:10].values
print(eksikveriler)
imputer =imputer.fit(eksikveriler[:,3:10])   #learning (learning what to replace missing values)
eksikveriler[:,3:10] =imputer.transform(eksikveriler[:,3:10])   #the part where the learned is applied (it replaces the missing values with the value)
print(eksikveriler)

#merging data and creating a dataframe (numpy arrays dataframe conversion)
#we got the date

date=veriler.iloc[:,1]
sonuc3=pd.DataFrame(data=date ,index=range(1567), columns= ['date'])

close=veriler.iloc[:,6]
sonuc=pd.DataFrame(data=close ,index=range(1567), columns= ['close'])
print(sonuc)

sonuc2=pd.DataFrame(data=eksikveriler ,index=range(1567),
                    columns=['open','high','low','close','Volume BTC','Volume USDT','tradecount'])
sonuc2.drop("close",axis=1,inplace=True)
print(sonuc2)

#dataframe merge operation
s=pd.concat([sonuc3,sonuc2],axis=1)
print(s)

# split data as train and test
x_train, x_test,y_train,y_test = train_test_split(sonuc2,sonuc,test_size=0.33, random_state=0)

#standardization
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#linear regression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


# predicting the accuracy score
score=r2_score(y_test,y_pred)
print('*************************************')
print('r2 score of Linear Regression is ',score)
print('mean_sqrd_error of Linear Regression is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of Linear Regression is==',np.sqrt(mean_squared_error(y_test,y_pred)))
print('*************************************')




#Decision Tree
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train,y_train)
y_pred2=r_dt.predict(x_test)

# predicting the accuracy score
score=r2_score(y_test,y_pred2)
print('r2 score of Decision Tree is ',score)
print('mean_sqrd_error of Decision Tree is==',mean_squared_error(y_test,y_pred2))
print('root_mean_squared error of Decision Tree is==',np.sqrt(mean_squared_error(y_test,y_pred2)))
print('*************************************')


#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
''' n_estimators ile kaç tane decision tree çizileceğini belirleriz.'''
y_train= y_train.values
rf_reg.fit(x_train,y_train.ravel())
y_pred3=rf_reg.predict(x_test)


# predicting the accuracy score
score=r2_score(y_test,y_pred3)
print('r2 score of Random Forest is ',score)
print('mean_sqrd_error of Random Forest is==',mean_squared_error(y_test,y_pred3))
print('root_mean_squared error of Random Forest is==',np.sqrt(mean_squared_error(y_test,y_pred3)))
print('*************************************')


# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.3
sfm = SelectFromModel(rf_reg, threshold=0.3)

# Train the selector
sfm.fit(x_train, y_train.ravel())



# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(x_train)
X_important_test = sfm.transform(x_test)

rfc_important=RandomForestRegressor(n_estimators = 10,random_state=0)
rfc_important.fit(X_important_train,y_train.ravel())

# guessing with new features
y_important_pred = rfc_important.predict(X_important_test)


score2=r2_score(y_test,y_important_pred)
print('r2 score of Random Forest with feature selection is ',score2)
print('mean_sqrd_error of Random Forest with feature selection is==',mean_squared_error(y_test,y_important_pred))
print('root_mean_squared error of Random Forest with feature selection is==',np.sqrt(mean_squared_error(y_test,y_important_pred)))
