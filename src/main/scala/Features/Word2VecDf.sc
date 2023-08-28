import org.apache.spark.ml.feature. Word2Vec

// Input data: Each row is a bag of words from a sentence or document
val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
).map(Tuple1.apply)).toDF("text")

// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec().
  setInputCol("text").
  setOutputCol("result").
  setVectorSize(3). // 3 dimension
  setMinCount(0) // 실제로는 min 값을 0으로 하지 않음. 유의미한 수준으로 설정

val model = word2Vec.fit(documentDF)

val result = model.transform(documentDF)
result.select("result").take(3).foreach(println)