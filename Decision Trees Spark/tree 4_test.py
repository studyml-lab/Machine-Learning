# --------------------
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# read data from csv files to dataframe
df2 = spark.read.csv('E:/kaggle/titanic/train.csv',header=True)
df2.count()

# ---------------------------------------

df2.printSchema()
df2.describe().show()
df2.groupby('Embarked').count().sort('count',ascending=False).show()
df3 = df2.select('Sex','Pclass','Survived','Embarked',df2.Fare.cast('double'),df2.Age.cast('double'))
df3.show()
df3.printSchema()
# -------------------------------------------

df3.describe().show()
from pyspark.ml.feature import Imputer

df3 = df3.na.fill({'Embarked':'S'})
df3.describe().show()
df3.groupby('Embarked').count().sort('count',ascending=False).first()[0]

imput1 = Imputer(inputCols=['Age','Fare'], outputCols=['Age1','Fare1'])
model111 = imput1.fit(df3)
model111.surrogateDF.show()

df3 = Imputer(inputCols=['Age','Fare'], outputCols=['Age1','Fare1']).fit(df3).transform(df3)


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df3).transform(df3)
df3.show()

df3 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df3)
df3.show()

# --------------------------------------------

df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)
df3 = OneHotEncoder(inputCol='Gender',outputCol='Gender1',dropLast=False).transform(df3)
df3.show()

# cast to double
df3 = df3.select(df3.Pclass.cast('double'),df3.Gender1,df3.Embarked2,df3.Survived.cast('double'))
df3.printSchema()


# Vector assembler

df3 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2'],outputCol='Features').transform(df3)
df3.show(truncate=False)

training = df3
training1 = df3

training.show(truncate=False,n=5)

# 1 choose approach
from pyspark.ml.classification import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived', seed=5000)

# 2 learning process - created a model

model2 = dt1.fit(training)
model2.depth
model2.numFeatures

training1 = model2.transform(training)
training1.show(5)

PredictionandLabels = training1.select(training1.prediction,training1.Survived).rdd
PredictionandLabels.collect()

from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
# metrics1 = BinaryClassificationMetrics(PredictionandLabels)
# (train score/train accuracy   --- )
# (train error = 1-train score ?)
metrics2 = MulticlassMetrics(PredictionandLabels)
metrics2.accuracy
metrics2.areaUnderPR
print(metrics2.confusionMatrix())

# ----------------------------------------------------------------------------
# CV / Parameter Tuning approach ---------------------------------------------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

paramGrid = ParamGridBuilder().addGrid(dt1.impurity,['entropy','gini']).addGrid(dt1.maxDepth,[2,3,4,5,6]).build()

evaluator1 = MulticlassClassificationEvaluator(predictionCol='prediction',
                                               labelCol='Survived',metricName='accuracy')

crossVal4 = CrossValidator(estimator=dt1,estimatorParamMaps=paramGrid,
                          evaluator=evaluator1, numFolds=10)


model23 = crossVal4.fit(df3)
model23.avgMetrics

# --------------------------------------------------------
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

#  df5_1.select('PassengerId','prediction').toPandas().to_csv('c:/test5.csv')


