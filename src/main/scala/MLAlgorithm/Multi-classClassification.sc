import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.util. MLUtils

// Load training data in LIBSVM format.
val sparkHome = System.getenv("SPARK_HOME")
val data = MLUtils.loadLibSVMFile(sc, s"${sparkHome}/data/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).

val splits = data.randomSplit(Array(0.6, 0.4), seed = 111)
val training = splits(0).cache()

val test = splits(1)

// Run training algorithm to build the model

val model = new LogisticRegressionWithLBFGS().
  setNumClasses(10). // 타겟 클래스 수를 세팅
  run(training)

// Compute raw scores on the test set.

val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.

val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy

println(s"Accuracy = $accuracy")

// Save and load model
val outputPath = s"${sparkHome}/data/output/model/scalalLogisticRegressionWithLBFGSModel"
model.save(sc, outputPath)
val sameModel = LogisticRegressionModel.load(sc, outputPath)