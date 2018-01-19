from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
sc = SparkContext()

# Several of the methods available in scala are currently missing from pyspark
# Load training data in LIBSVM format
data = MLUtils.loadLibSVMFile(sc, "F:/spark-2.2.1-bin-hadoop2.7/data/mllib/sample_binary_classification_data.txt")

# Split data into training (60%) and test (40%)
training, test = data.randomSplit([0.6, 0.4], seed=11)
training.cache()

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

# Run training algorithm to build the model
model = LogisticRegressionWithLBFGS.train(training)



# Compute raw scores on the test set
predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

predictionAndLabels.collect()

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)
metrics1 = MulticlassMetrics(predictionAndLabels)


metrics1.accuracy

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
