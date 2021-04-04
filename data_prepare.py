#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType

spark = SparkSession.builder.appName("filkin_spark").getOrCreate()

habr_data = spark.read\
    .option("header", True)\
    .option("inferSchema", True)\
    .csv("/user/admin/habr_data.csv")\
    .na.drop("any")\
    .cache()

habr_data.printSchema()
habr_data.show(1, False)

# чтобы было, что джойнить, придумываем пользователям ID и выносим "author_type" и "author_name"

habr_users = habr_data\
    .select('author_name', 'author_type')\
    .distinct()\
    .withColumn('author_id', F.monotonically_increasing_id())

habr_users = habr_users.select('author_id', 'author_name', 'author_type')

habr_data = habr_data.join(habr_users, 'author_name').drop('author_name', 'author_type')

# запишем habr_users в отдельную таблицу в Cassandra

habr_users.write \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="habr_users", keyspace="filkin") \
    .mode("append") \
    .save()

# делаем из данных два датасета, один для обучения, второй для кафки (который будем читать и "предсказывать")

habr_learn, habr_kafka = habr_data.randomSplit([.9, .1], seed=42)

columns_to_drop = ['h3_count', 'i_count', 'spoiler_count', 'positive_votes',
                   'negative_votes', 'rating', 'bookmarks', 'views', 'comments']

habr_kafka = habr_kafka.drop(*columns_to_drop)

# записываем данные для предсказания в кафку

habr_kafka.selectExpr("CAST(null AS STRING) as key", "CAST(to_json(struct(*)) AS STRING) as value") \
        .write \
        .format("kafka") \
        .option("topic", "filkin_habr") \
        .option("kafka.bootstrap.servers", "bigdataanalytics-worker-0.novalocal:6667") \
        .option("checkpointLocation", "my_kafka_checkpoint") \
        .save()


