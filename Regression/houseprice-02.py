from sklearn import linear_model
import pandas as pd
from sklearn import model_selection
from sklearn import tree


#1. Undestand Business Problem
##
##

# 2. Data Collection
##
##
house_train = pd.read_csv('e:/kaggle/homeprices/train.csv')
house_test = pd.read_csv('e:/kaggle/homeprices/test.csv')

# 3. EDA (EDA report)
##
## 
house_train.shape
house_train.info()

house_test.shape
house_test.info()

# 4.Preprocessing
## Categoric to continuous
##

# FE
dims= ['LotArea']
X_train = house_train[dims]
y_train = house_train['SalePrice']

# MB

# MB-1 : Choose your modelling approach
lr1 = linear_model.LinearRegression()


lr1.fit(X_train,y_train)

# train score
lr1.score(X_train,y_train)

lr2 = linear_model.Ridge()
lr2 = linear_model.Lasso()
lr4 = linear_model.RidgeCV()
dt5 = tree.DecisionTreeRegressor

# MB-2 : fit call is actual learning prrocess
# MB -2.1 : ME

# CV score
model_selection.cross_val_score(lr1,X_train,y_train,cv=10)



lr2.score(X_train,y_train)

lr2.fit(X_train,y_train)
lr2.coef_
lr2.intercept_
lr2.fit_intercept

# 7 ME
# Evaluation Strategies
1
2
3
4
5


# Manual Evaluation
# Automation Evaluation


# predictions on test data and submit it to kaggle.
X_test = house_test[dims]
house_test['SalePrice'] = lr1.predict(X_test)

house_test.to_csv('house1.csv', columns=['Id','SalePrice'],index=False)

# End



