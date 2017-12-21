from sklearn import linear_model
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
import numpy as np


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
y_train = np.log(house_train['SalePrice'])

# MB

# MB-1 : Choose your modelling approach
# mylr = linear_model.LinearRegression()
# lr2 = linear_model.Ridge()
mylr = linear_model.Lasso(max_iter=50)
# lr4 = linear_model.RidgeCV()
# dt5 = tree.DecisionTreeRegressor


mylr.fit(X_train,y_train)

mylr.coef_
mylr.intercept_
mylr.n_iter_

# estimators will have scoring function (score), they will predict y_pred and calculate score or loss by using y_train and y_pred

# Resubstitution strategy, y_pred, explained variance
mylr.score(X_train,y_train)
--------------------------------------

y_pred = mylr.predict(X_train)

metrics.explained_variance_score(y_train,y_pred)

math.sqrt(metrics.mean_squared_error(y_train,y_pred))
------------------------------

CV approach  (cross valiation approach)

model_selection.cross_val_score(mylr, X_train, y_train, cv=10).mean()

# math.sqrt(0.1536)
--------------------
math.sqrt(5867752122.5090752)


# train score
lr1.score(X_train,y_train)

# MB-2 : fit call is actual learning prrocess
# MB -2.1 : ME





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
# OR alternatively plan for model deployment for realtime or batch predictions
X_test = house_test[dims]
house_test['SalePrice'] = lr1.predict(X_test)

house_test.to_csv('house1.csv', columns=['Id','SalePrice'],index=False)

# End



