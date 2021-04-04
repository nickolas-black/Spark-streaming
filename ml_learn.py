#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

spark = SparkSession.builder.appName("filkin_spark").getOrCreate()

# подготавливаем датасет для обучения, приводим rating к IntegerType

habr_learn = habr_learn.withColumn("rating", F.expr("CAST(rating as INTEGER)"))\
                                   .na.drop("any") \
                                   .cache()

# Пишем функцию которая определяет категорию статьи ( A, B, C, D) на основе рейтинга,
# и добавляем к датасету категориальную фичу "rating_class".

def rating_to_class(rating):
    if rating < 10:
        return 'D'
    elif rating < 20:
        return 'C'
    elif rating < 60:
        return 'B'
    return 'A'

udf_rating_to_class = F.udf(rating_to_class, StringType())

habr_learn = habr_learn.withColumn('class', udf_rating_to_class('rating'))

# словарь для преобразования класса в число, т.к. модель понимает только числа

class_to_int = {'A': 1,
                'B': 2,
                'C': 3,
                'D': 4
                }

udf_class_to_int = F.udf(lambda x: class_to_int[x], IntegerType())

# Строим модель логистической регрессии (one vs all) для классификации статей по рассчитанным классам.
# Для упрощения - учим только по полю "ten_most_common_words", в теории можно учить по множеству полей

train, test = habr_learn.randomSplit([.8, .2], seed=42)

train = train.withColumn('class', udf_class_to_int('class'))
test = test.withColumn('class', udf_class_to_int('class'))

tokenizer = Tokenizer(inputCol="ten_most_common_words", outputCol="10_words_array")

hashingTF = HashingTF(inputCol="10_words_array", outputCol="rawFeatures", numFeatures=200000)

idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(featuresCol='features', labelCol='class', predictionCol='prediction',
                        maxIter=10, tol=1E-6, fitIntercept=True)
ovr = OneVsRest(featuresCol='features', labelCol='class', predictionCol='prediction', classifier=lr)

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, ovr])

model = pipeline.fit(train)

model.write().overwrite().save("habr_LR_model")


