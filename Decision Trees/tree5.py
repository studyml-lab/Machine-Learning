# if you need pydot - install using below command
# pip install pydot
# install graphviz 'graphviz-2.38.msi' and set add install directory to path
# here is the path: http://graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
# here is the alternate path : http://graphviz.org/Download_windows.php
# add 'C:\Graphviz2.38\bin' or install location to 'Path' environment (system variables), 
# might need admin privileges

import pandas as pd
from sklearn import tree
import pydot
import io
import os
from sklearn import model_selection, metrics
# if you want to do naive bayes
# from sklearn.naive_bayes import GaussianNB
# ----------------------------
# Data collection

os.chdir('e:/titanic')

titanic_train = pd.read_csv("train_kaggle.csv")
print(type(titanic_train))

# ----------------------------

#explore the dataframe  (EDA)
titanic_train.shape
titanic_train.info()


#  Data Preprocessing --------------------------

titanic_train.loc[titanic_train['Age'].isnull() == True  , 'Age'] = titanic_train['Age'].mean()
titanic_train.loc[titanic_train['Embarked'].isnull() == True  , 'Embarked'] = 'S'

titanic_train = pd.get_dummies(titanic_train,columns=['Sex','Embarked','Pclass'])

# F E -----------------------------


# approach 1 = select those column we feel important


cols_mb = ['Age','SibSp','Sex_female','Sex_male','Fare', 'Embarked_C','Embarked_Q','Embarked_S','Pclass_1','Pclass_2','Pclass_3']

X_train = titanic_train[cols_mb]
y_train = titanic_train['Survived']

# approach 2 -> drop those columns that are not important

# X_train = titanic_train.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)

# -----------------------------
# MB phase
# 1 model object

tree_model1 = tree.DecisionTreeClassifier()



# nb_model1 = GaussianNB()
# logi_reg = 
# knn_model = 

# prob_model1 = ------------------------------------------

# 2 learning process

# Model evaluation: CV approach
# model_selection.cross_val_score(tree_model1,X_train,y_train,cv=10).mean()   #-- 0.684


# Model evaluation: GridSearchCV approach:

grid_estimator = model_selection.GridSearchCV(tree_model1,grid_dt,cv=10)

grid_dt = {'max_depth':[5,10],
           'criterion':['entropy','gini'],
           'min_samples_split':[3,5,8]           
           }

grid_estimator.fit(X_train,y_train)
grid_estimator.best_params_
grid_estimator.grid_scores_
grid_estimator.best_score_

grid_estimator.best_estimator_.score(X_train,y_train)


mytree = grid_estimator.best_estimator_


#   --- built 10 trees with 90% of data 10%

#------------------------

# ------------------------------

# Learn tree model
tree_model1.fit(X_train, y_train)
tree_model1.n_classes_    # classes in survived column
tree_model1.n_features_   # no of features used for MB

# Resubstitution approach
#  -- built model with 100% train data, validated with 100%

y_pred = tree_model1.predict(X_train)
metrics.confusion_matrix(y_train,y_pred)
#  874/891
tree_model1.score(X_train,y_train)   # --  0.69
metrics.accuracy_score(y_train,y_pred)


# 3 predict 
titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test.info()
X_test =  titanic_test[['Pclass','SibSp']]
titanic_test['Survived'] = tree_model1.predict(X_test)


titanic_test.to_csv('submission5.csv', columns=['PassengerId','Survived'],index=False)
# ---------------------------------------------



#visualize the decision tree
dot_data = io.StringIO() 
tree.export_graphviz(tree_model1, out_file = dot_data, feature_names = X_train.columns, filled=True, rounded=True,  special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("e:/titanic/decision-tree23.pdf")
