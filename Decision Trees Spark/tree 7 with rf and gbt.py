# --------------------
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# read data from csv files to dataframe
df2 = spark.read.csv('E:/kaggle/titanic/train.csv',header=True)
df2.count()

# ---------------------------------------

df2.printSchema()
df2.describe().show()

embarked_mode = df2.groupby('Embarked').count().sort('count',ascending=False).first()[0]
df2 = df2.na.fill({'Embarked':embarked_mode})

df3 = df2.select('Sex',df2.Pclass.cast('double'),df2.Survived.cast('double'),'Embarked',df2.Fare.cast('double'),df2.Age.cast('double'))

from pyspark.ml.feature import Imputer

df3 = Imputer(inputCols=['Age','Fare'], outputCols=['Age1','Fare1']).fit(df3).transform(df3)

df3.show(3)
#--------------------------------


# df3 = df2.select('Sex',df2.Pclass.cast('double'),df2.Survived.cast('double'),'Embarked',df2.Fare.cast('double'),df2.Age.cast('double'))

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

df3 = StringIndexer(inputCol='Embarked',outputCol='Embarked1').fit(df3).transform(df3)
df3.show()

df3 = OneHotEncoder(inputCol='Embarked1',outputCol='Embarked2',dropLast=False).transform(df3)
df3.show()

# --------------------------------------------

df3 = StringIndexer(inputCol='Sex',outputCol='Gender').fit(df3).transform(df3)
df3 = OneHotEncoder(inputCol='Gender',outputCol='Gender1',dropLast=False).transform(df3)
df3.show(5)

# cast to double
#df3 = df3.select(df3.Pclass.cast('double'),df3.Gender1,df3.Embarked2,df3.Survived.cast('double'))
#df3.printSchema()


# Vector assembler

df3 = VectorAssembler(inputCols=['Pclass','Gender1','Embarked2','Fare1','Age1'],outputCol='Features').transform(df3)
df3.show(truncate=False)

# data processing complete---

# 6 .Model building
training = df3
training1 = df3
training.show(truncate=False,n=5)

# 1 choose approach

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier


gbt1 = GBTClassifier(featuresCol='Features',labelCol='Survived',maxIter=100)

gbt_model1 = gbt1.fit(training)

training38 = gbt_model1.transform(training)
PredictionsandLabels = training38.select('prediction','Survived').rdd

from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

metrics1 = MulticlassMetrics(PredictionsandLabels)
metrics1.accuracy






rf1 = RandomForestClassifier(featuresCol='Features',labelCol='Survived',numTrees=100)

rf_model1 = rf1.fit(training)
rf_model1.getNumTrees
rf_model1.numClasses

print(rf_model1.featureImportances)

training20 = rf_model1.transform(training)




from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
dt1 = DecisionTreeClassifier(featuresCol='Features',labelCol='Survived', maxDepth=30)

from pyspark.mllib.classification import Random
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier

rf1 = RandomForestClassifier(featuresCol='Features',labelCol='Survived')
rf_model1 = rf1.fit(training)


training2_1 = rf_model1.transform(training)

PredictionsandLabels = training2_1.select('prediction','Survived').rdd

PredictionsandLabels.collect()

# 2 learning process - created a model

model22 = dt1.fit(training)
model22.depth
model22.numFeatures

model22.save('E:/kaggle/model22')

model120 = DecisionTreeClassificationModel()
model122 = model120.load('E:/kaggle/model22')

training4 = model122.transform(training)

training4.show(3)
model23 = 

training2 = model22.transform(training)

PredictionsandLabels = training2.select('prediction','Survived').rdd

PredictionsandLabels.collect()
# --------------------------------------------------------------

#Resubstitution approach
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

metrics1 = MulticlassMetrics(PredictionsandLabels)
metrics1.accuracy

# --------------------------------------------------------------------------

# 1 step calculate cv score for 1 model

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


evaluator2 = BinaryClassificationEvaluator(labelCol='Survived',rawPredictionCol='prediction')
paramGrid = ParamGridBuilder().addGrid(rf1.numTrees,[20,30,40,50,60]).build()

crossval1 = CrossValidator(estimator=rf1,estimatorParamMaps=paramGrid,
               evaluator=evaluator1,
               numFolds=10)

crossval2 = CrossValidator(estimator=rf1,estimatorParamMaps=paramGrid,
               evaluator=evaluator2,
               numFolds=10)

model27 = crossval2.fit(training)

model27.bestModel.

model25 = crossval1.fit(training)
model26 = crossval1.fit(training)

model27.avgMetrics


model27.bestModel

training2 = model27.transform(training)

# ------------------------------------------------------------------------------------

training2.show(3)

# right parameters -  (parameter tuning, model selection).







# model22.save('E:/kaggle/titanic/dt_model10')

# -----------------------------------------------------------------------
# Step 8 for deployment --- save and load model for operationalization purpose -
from pyspark.ml.classification import DecisionTreeClassificationModel
model3 = DecisionTreeClassificationModel()
model3 = model3.load('E:/kaggle/titanic/dt_model10')

model3.depth
model3.numFeatures

from pyspark.ml.linalg import Vectors
predict_df = spark.createDataFrame([(1,Vectors.dense(1.0,0.0,1.0,0.0,1.0,0.0))],['index','Features'])
predict_df.show()
model3.transform(predict_df).select('prediction').first()[0]

# ------------------------------------------------------------------------

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


