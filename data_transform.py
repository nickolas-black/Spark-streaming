#export SPARK_KAFKA_VERSION=0.10
#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("filkin_spark").getOrCreate()

kafka_brokers = "bigdataanalytics-worker-1.novalocal:6667"

# читаем из кафки поступающие данные о "новых" статьях

habr = spark.readStream. \
    format("kafka"). \
    option("kafka.bootstrap.servers", kafka_brokers). \
    option("subscribe", "filkin_habr"). \
    option("startingOffsets", "earliest"). \
    option("maxOffsetsPerTrigger", "1"). \
    load()

schema = StructType() \
    .add("link", StringType()) \
    .add("title", StringType()) \
    .add("published_date", StringType()) \
    .add("published_time", StringType()) \
    .add("modified_date", StringType()) \
    .add("modified_time", StringType()) \
    .add("description", StringType()) \
    .add("image", StringType()) \
    .add("article_categories", StringType()) \
    .add("href_count", StringType()) \
    .add("img_count", StringType()) \
    .add("tags", StringType()) \
    .add("text_len", StringType()) \
    .add("lines_count", StringType()) \
    .add("sentences_count", StringType()) \
    .add("first_5_sentences", StringType()) \
    .add("last_5_sentences", StringType()) \
    .add("max_sentence_len", StringType()) \
    .add("min_sentence_len", StringType()) \
    .add("mean_sentence_len", StringType()) \
    .add("median_sentence_len", StringType()) \
    .add("tokens_count", StringType()) \
    .add("max_token_len", StringType()) \
    .add("mean_token_len", StringType()) \
    .add("median_token_len", StringType()) \
    .add("alphabetic_tokens_count", StringType()) \
    .add("words_count", StringType()) \
    .add("words_mean", StringType()) \
    .add("ten_most_common_words", StringType()) \
    .add("author_id", StringType())

habr_value = habr.select(F.from_json(F.col("value").cast("String"), schema).alias("value"), "offset")
habr_articles = habr_value.select(F.col("value.*"), "offset")

def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='{0} seconds'.format(freq) ) \
        .options(truncate=False) \
        .start()

s = console_output(habr_articles, 5)
s.stop()

# в полученных данных нет имени и типа пользователя. Запросим эти данные
# из Кассандры, таблица habr_users, для дальнейшего джойна по id

cassandra_users = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="habr_users", keyspace="filkin" ) \
    .load()

cassandra_users.show()

# загружаем модель из hdfs

model = PipelineModel.load("habr_LR_model")

# словарь для преобразования числа в класс, т.к. модель предсказывает только числа

int_to_class = {1: 'A',
                2: 'B',
                3: 'C',
                4: 'D'
                }

udf_int_to_class = F.udf(lambda x: int_to_class[x], StringType())

# логика foreachBatch

def writer_logic(df, epoch_id):
    df.persist()
    # сразу засовываем полученные из кафки данные в модель, в них есть все нужное
    habr_predict = model.transform(df).withColumn('class', udf_int_to_class('prediction'))
    # берем только нужное
    required_attributes = ['link', 'title', 'published_date', 'published_time',
                           'description', 'image', 'article_categories', 'tags',
                           'class', 'author_id']
    habr_prepared = habr_predict.select(*required_attributes)
    # к данным хотим добавить имя автора и его тип, достаем из ДФ author_id и по ним заберем данные из кассандры
    authors_list_batch = habr_prepared.select('author_id').distinct().collect()
    authors_list = map(lambda x: str(x.__getattr__('author_id')) , authors_list_batch)
    where_string = " author_id = " + " or author_id = ".join(authors_list)
    authors_cassandra = cassandra_users.where(where_string)
    # объединяем данные и записываем в Кассандру
    habr_final = habr_prepared.join(authors_cassandra, 'author_id')
    print("Here is what I'm writing to the Cassandra:")
    habr_final.show()
    habr_final.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="habr_recs", keyspace="filkin") \
        .mode("append") \
        .save()
    df.unpersist()


#связываем источник Кафки и foreachBatch функцию
stream = habr_articles \
    .writeStream \
    .trigger(processingTime='30 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/habr_checkpoint")

#поехали
s = stream.start()

s.stop()

