import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
val sparkHome = System.getenv("SPARK_HOME")
val input = sc.textFile(s"${sparkHome}/data/mllib/sample_lda_data.txt").
  map(line => line.split(" ").toSeq) // one line as one document

val word2vec = new Word2Vec()

val model = word2vec.fit(input)

val synonyms = model.findSynonyms("1", 5)
for((synonym, cosineSimilarity) <- synonyms) {
  println(s"$synonym $cosineSimilarity")
}

// Save and load model
val modelPath = s"${sparkHome}/data/output/model/Word2VecRdd"
model.save(sc, modelPath)
val sameModel = Word2VecModel.load(sc, modelPath)