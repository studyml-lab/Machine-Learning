# if you need pydot - install using below command
# pip install pydot
# install graphviz 'graphviz-2.38.msi' and set add install directory to path
# add 'C:\Graphviz2.38\bin' or install location to 'Path' environment (system variables), 
# might need admin privileges

import pandas as pd
from sklearn import tree
import pydot
import io
import os
from sklearn import model_selection, metrics
# ----------------------------
# Data collection

os.chdir('e:/titanic')

titanic_train = pd.read_csv("train.csv")
print(type(titanic_train))

# ----------------------------

#explore the dataframe  (EDA)
titanic_train.shape
titanic_train.info()

# F E -----------------------------

X_train = titanic_train[['Pclass', 'SibSp']]
y_train = titanic_train['Survived']

# -----------------------------
# MB phase
# 1 model object

tree_model1 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6 )
tree_model2 = tree.DecisionTreeClassifier(criterion='gini',max_depth=3)

# prob_model1 = ------------------------------------------

# 2 learning process

model_selection.cross_val_score(tree_model1,X_train,y_train,cv=10).mean()   -- 0.684
#   --- built 10 trees with 90% of data 10%

model_selection.cross_val_score(tree_model2,X_train,y_train,cv=10).mean()

tree_model1.fit(X_train, y_train)
tree_model1.n_classes_    # classes in survived column
tree_model1.n_features_   # no of features used for MB


tree_model1.score(X_train,y_train)   --  0.69
#  -- built model with 100% train data, validated with 100%


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
graph.write_pdf("decision-tree22.pdf")
