# Please run this program from anaconda prompt (command line)
# python "Program path and name"
# python "e:\studyml-lab\Machine-Learning\Deployment\model deploy and run service.py"

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# import numpy as np
from flask import Flask, jsonify, request

from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.linalg import Vectors


model40 = DecisionTreeClassificationModel()
model141 = model40.load('E:/kaggle/model22')
model141.depth
model141.numFeatures

app = Flask(__name__)
@app.route('/api',methods=('GET','POST'))

def make_predict():
    print('hi, good morning ... ')
    
    data = request.get_json(force=True)
    print(data)
    predict_df = spark.createDataFrame([(1,Vectors.dense(data))],['index','Features'])
    predict_df.show()
    output = model141.transform(predict_df).select('prediction').first()[0]
    print(output)

    return jsonify('Survived' if output==1 else 'Not Survived')

if __name__ == '__main__':
    app.run(port = 9000, debug = True)
    
 