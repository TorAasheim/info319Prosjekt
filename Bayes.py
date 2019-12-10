from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import CountVectorizer
import plotly.graph_objects as go
import pandas as pd
import datetime



# Before running, change spark-default.conf file so that   spark.driver.memory 5g is not commented out, if the file is a template, copy the file and remote.template fileending
#This is so that the JVM has enough memory to run the program. the file is located in the spark folder/conf
spark = SparkSession.builder.appName("Bayes").getOrCreate()

#Load training data, select right columns and change features to ints
train = spark.read.format("csv").option("header", "true").load("training.csv")
drop_list = ['ItemID', 'SentimentSource', ]
train = train.select([column for column in train.columns if column not in drop_list])
train = train.withColumn("label", train["label"].cast("TINYINT"))

#removes null rows
train = train.na.drop()

#Load test data
test = spark.read.format("csv").option("header", "true").option("delimiter", ";").load("cloud.csv")

#tokenizer
trainTokenizer = Tokenizer(inputCol="SentimentText;;;;;;;;;;;;;;;;;;;;;;;;", outputCol="words")
trainCountTokens = udf(lambda words: len(words), IntegerType())
train = trainTokenizer.transform(train)

testTokenizer = Tokenizer(inputCol="text", outputCol="words")
testCountTokens = udf(lambda words: len(words), IntegerType())
test = testTokenizer.transform(test)

#Remove stopwords
trainRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
train = trainRemover.transform(train)

testRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
test = testRemover.transform(test)

# fit CountVectorizerModel
trainCv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=3, minDF=2.0)
trainModel = trainCv.fit(train)
train = trainModel.transform(train)

testCv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=3, minDF=2.0)
testModel = testCv.fit(test)
test = testModel.transform(test)

#create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

#train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

#Remove unwanted columns before writing to file
predictions = predictions.drop('words')
predictions = predictions.drop('filtered')
predictions = predictions.drop('features')
predictions = predictions.drop('rawPrediction')
predictions = predictions.drop('probability')
predictions = predictions.drop('text')


#Transform spark dataframe into pandas dataframe, change time column to pandas datetime datatype for visualization
predictions.createOrReplaceTempView("predictions")
viz = predictions.select("*")
viz = viz.toPandas()
viz['time'] = pd.to_datetime(viz['time'], errors='coerce')

#Drops the rows witch dont fit the datatime structure
viz = viz.dropna(subset=['time'])
viz.index = viz['time']

#group by month and takes the mean of the predictions
viz = viz.resample('M').mean()

#Dates used for the xAxis in the graph
dates = [datetime.datetime(year=2015, month=1, day=31),
     datetime.datetime(year=2015, month=2, day=28),
     datetime.datetime(year=2015, month=3, day=31),
     datetime.datetime(year=2015, month=4, day=30),
     datetime.datetime(year=2015, month=5, day=31),
     datetime.datetime(year=2015, month=6, day=30),
     datetime.datetime(year=2015, month=7, day=31),
     datetime.datetime(year=2015, month=8, day=31),
     datetime.datetime(year=2015, month=9, day=30),
     datetime.datetime(year=2015, month=10, day=31),
     datetime.datetime(year=2015, month=11, day=30),
     datetime.datetime(year=2015, month=12, day=31),
     datetime.datetime(year=2016, month=1, day=31),
     datetime.datetime(year=2016, month=2, day=29),
     datetime.datetime(year=2016, month=3, day=31),
     datetime.datetime(year=2016, month=4, day=30),
     datetime.datetime(year=2016, month=5, day=31)]

#Creates a figure that is written to png
fig = go.Figure([go.Scatter(x=dates, y=viz['prediction'])])
fig.update_layout(
    title="All predictions averaged on each month",
    xaxis_title="Date",
    yaxis_title="Prediction(1= positive, 0= negative)")

fig.write_image("fig1.png")
predictions.write.csv("predictions")