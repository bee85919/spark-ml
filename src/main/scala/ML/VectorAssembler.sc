import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "time", "device", "userFeatures", "clicked")

val assembler = new VectorAssembler().
  setInputCols(Array("time", "device", "userFeatures")).
  setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features", "clicked").first())