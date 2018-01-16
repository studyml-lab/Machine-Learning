# if you need pydot - install using below command
# pip install pydot
# install graphviz 'graphviz-2.38.msi' and set add install directory to path
# add 'C:\Graphviz2.38\bin' to 'Path' environment (system variables), might need admin privileges


import pandas as pd
from sklearn import tree
import pydot   # pydot and graviz 
import io
import os
# ----------------------------
# Data collection

os.chdir('e:/titanic')

titanic_train = pd.read_csv("train.csv")
print(type(titanic_train))

# ----------------------------

#explore the dataframe  (EDA)
titanic_train.shape
titanic_train.info()

titanic_train.Sex.value_counts()

# Data preprocessing
titanic_train.loc[titanic_train.Sex == 'male','Sex'] = 2
titanic_train.loc[titanic_train.Sex == 'female','Sex'] = 1

# F E -----------------------------

X_train = titanic_train[['Pclass','SibSp','Parch','Sex','Fare']]
y_train = titanic_train['Survived']

# ---------MB phase --------------------
# MB phase
# 1 model object

tree_model1 = tree.DecisionTreeClassifier(criterion='entropy')

# 2 learning process

tree_model1.fit(X_train, y_train)

tree_model1.n_classes_    # classes in survived column
tree_model1.classes_
tree_model1.n_features_   # no of features used for MB



tree_model1.score(X_train, y_train)

y_train_pred = tree_model1.predict(X_train)

from sklearn import metrics
metrics.confusion_matrix(y_train,y_train_pred)

#  accuracy --- Model error -- 


# 3 predict 
titanic_test = pd.read_csv('test.csv')

titanic_test.shape
titanic_test.info()

X_test =  titanic_test[['Pclass','SibSp','Parch','Sex']]

titanic_test['Survived'] = tree_model1.predict(X_test)

titanic_test[['PassengerId','Survived']]

titanic_test.to_csv('submission122.csv', columns=['PassengerId','Survived'],index=False)
# ---------------------------------------------


#visualize the decision tree
dot_data = io.StringIO() 
tree.export_graphviz(tree_model1, out_file = dot_data, feature_names = X_train.columns, filled=True, rounded=True,  special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("decision-tree52.pdf")
