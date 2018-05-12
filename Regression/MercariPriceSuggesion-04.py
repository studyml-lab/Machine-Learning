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
house_train = pd.read_csv('e:/kaggle/Mercari price suggesion/train.tsv',sep='\t')
house_test = pd.read_csv('e:/kaggle/Mercari price suggesion/test.tsv',sep='\t')

# 3. EDA (EDA report)
##
## 
house_train.shape
house_train.info()
house_train.head(5)


house_test.shape
house_test.info()

# 4.Preprocessing
## Categoric to continuous
##

cat_to_cont = ['MSSubClass','Street']

house_train = pd.get_dummies(house_train,columns=cat_to_cont)

house_train.columns

# house_train['GarageQual']
mappings = { 'Po' : 1,
            'Fa' : 2,
            'TA' : 3,
            'Gd' : 4,
            'Ex' : 5
          }

#functional style of programming

features1 = ['ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual', 'HeatingQC',
             'GarageQual','GarageCond','FireplaceQu']

for f1 in features1:
    house_train[f1] = house_train[f1].map(mappings)
    house_train.loc[house_train[f1].isnull()==True,f1]= 0

# FE
dims= ['LotArea','OverallQual','OverallCond','MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45',
       'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75',
       'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120',
       'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'Street_Grvl',
       'Street_Pave','ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual', 'HeatingQC',
             'GarageQual','GarageCond','FireplaceQu']

X_train = house_train[dims]
y_train = np.log(house_train['SalePrice'])

X_train.info()

# MB

# MB-1 : Choose your modelling approach
# mylr = linear_model.LinearRegression()
# lr2 = linear_model.Ridge()
mylr = linear_model.Lasso()
# lr4 = linear_model.RidgeCV()
# dt5 = tree.DecisionTreeRegressor

#---------------------------------------
#grid approach
mygrid1 = {'alpha' : np.arange(200,300,10)
        }

mylr_grid_estimator = model_selection.GridSearchCV(mylr,mygrid1, cv=10)

mylr_grid_estimator.fit(X_train,y_train)

mylr_grid_estimator.grid_scores_
mylr_grid_estimator.best_estimator_

mylr_grid_estimator.best_estimator_.coef_
mylr_grid_estimator.best_estimator_.intercept_

mylr_grid_estimator.best_estimator_.
# ----------------------------------------------

mylr.fit(X_train,y_train)

mylr.coef_
mylr.intercept_
mylr.n_iter_


# estimators will have scoring function (score), they will predict y_pred and calculate score or loss by using y_train and y_pred

# Resubstitution strategy, y_pred, explained variance
# mylr.score(X_train,y_train)
# --------------------------------------

y_pred = mylr.predict(X_train)


#metrics.explained_variance_score(y_train,y_pred)

math.sqrt(metrics.mean_squared_error(y_train,y_pred))

------------------------------

CV approach  (cross valiation approach)

model_selection.cross_val_score(mylr_grid_estimator.best_estimator_, X_train, y_train, cv=10,

            scoring=metrics.make_scorer(metrics.mean_squared_error)).mean()

import math
math.sqrt(1675578741.8152092)

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



