# --------------------
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# read data from csv files to dataframe
df2 = spark.read.csv('E:/kaggle/titanic/train_kaggle.csv',header=True)
df2.count()

# ---------------------------------------

df2.printSchema()
df3 = df2.select('Sex','Pclass','Survived','Embarked')
df3.show()
df3.printSchema()
# -------------------------------------------
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df3).transform(df3)
df3.show()

df3 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df3)
df3.show()

# --------------------------------------------

df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)
df3 = OneHotEncoder(inputCol='Gender',outputCol='Gender1',dropLast=False).transform(df3)
df3.show()


df3 = df3.select(df3.Pclass.cast('double'),df3.Gender1,df3.Embarked2,df3.Survived.cast('double'))
df3.printSchema()


# Vector assembler

df3 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2'],outputCol='Features').transform(df3)
df3.show(truncate=False)
#
# 1 choose approach
from pyspark.ml.classification import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived')

# 2 learning process - created a model
model2 = dt1.fit(df3)
model2.depth
model2.numFeatures

# 3 get predictions

df5 = spark.read.csv('E:/kaggle/titanic/test.csv',header=True).select('PassengerId','Sex','Pclass','Embarked')

df5 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df5).transform(df5)
df5.show()

df5 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df5)
df5.show()

# --------------------------------------------

df5 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df5).transform(df5)
df5 = OneHotEncoder(inputCol='Gender',outputCol='Gender1',dropLast=False).transform(df5)
df5.show()


df5 = df5.select(df5.Pclass.cast('double'),df5.Gender1,df5.Embarked2,df5.PassengerId)
df5.printSchema()

# Vector assembler

df5 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2'],outputCol='Features').transform(df5)
df5.show(truncate=False)


df5_1 = model2.transform(df5)
df5_1.show()

df5_1.select('PassengerId','prediction').coalesce(1).write.csv('c:/test5.csv')


