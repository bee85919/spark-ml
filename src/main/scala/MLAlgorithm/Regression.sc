import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector

import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

// Load and parse the data
val sparkHome = System.getenv("SPARK_HOME")
val data = sc.textFile(s"${sparkHome}/data/mllib/ridge-data/lpsa.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}.cache()

// Building the model
val algorithm = new LinearRegressionWithSGD()
val model = algorithm.run(parsedData)

// Evaluate model on training examples and compute training error
val valuesAndPreds = parsedData.map { point =>
  val prediction = model. predict(point.features)
  (point.label, prediction)
}

// 정답과 에러의 차이, 작을수록 좋다.
val MSE = valuesAndPreds.map{ case(v, 0) => math.pow((v - 0), 2) }.mean() // 평균
println("training Mean Squared Error =" + MSE)

// Save and load model
val outputPath = s"${sparkHome}/data/output/model/scalalinearRegressionWithSGDModel"
model.save(sc, outputPath)
val sameModel = LinearRegressionModel.load(sc, outputPath)