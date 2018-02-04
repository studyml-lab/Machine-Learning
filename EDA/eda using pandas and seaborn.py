import seaborn as sns
import pandas as pd
titanic_train = pd.read_csv('E:/kaggle/titanic/train.csv')

#categorical columns: statistical EDA
pd.crosstab(index=titanic_train["Survived"], columns="count")
pd.crosstab(index=titanic_train["Pclass"], columns="count")  
pd.crosstab(index=titanic_train["Sex"],  columns="count")

#categorical columns: visual EDA
sns.countplot(x='Sex',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)

#continuous features: statistical EDA
titanic_train['Fare'].describe()
titanic_train[['Age','Fare']].describe()

#continuous features: visual EDA

sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], kde=False)
sns.distplot(titanic_train['Fare'], bins=20, rug=False, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)

#bivariate relationships(c-c): statistical EDA 
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Sex'])
pd.crosstab(index=titanic_train['Survived'], columns=titanic_train['Pclass'], margins=True)

# factorplot 
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)


#bivariate relationships(n-c): visual EDA 
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=10).map(sns.distplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.boxplot, "Fare").add_legend()

titanic_train.loc[titanic_train['Age'].isnull() == True, 'Age'] = titanic_train['Age'].mean()

sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "Age").add_legend()


# Multi variate 
# FacetGrid creates a grid for rows and columns and then analyses the values in map function
# another option is pair plot   --   sns.pairplot(.....)

sns.FacetGrid(titanic_train, row="Sex", col="Pclass").map(sns.countplot, "Survived")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.distplot, "Fare")
sns.FacetGrid(titanic_train, row="Survived", col="Sex").map(sns.distplot, "Age")
sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()
