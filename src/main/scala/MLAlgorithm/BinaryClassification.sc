import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util. MLUtils

// Load training data in LIBSVM format.
val sparkHome = System.getenv("SPARK_HOME")
val data = MLUtils.loadLibSVMFile(sc, s"${sparkHome}/data/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).
// 실제로는 train 의 비중을 더 높인다.
val splits = data.randomSplit(Array(0.6, 0.4), seed = 111)

val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
// Stochastic Gradient Descent

// Clear the default threshold.
model.clearThreshold() //  ROC score 로 구분기준을 사용하기위해 클리어

// Compute raw scores on the test set.

val scoreAndLabels = test.map { point => //labeldpoint
  val score = model.predict(point.features) // feature 만 뽑은
  (score, point.label) // 예측결과와 실제 값만 반환
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC =" + auROC)

// Save and load model
val outputPath = s"${sparkHome}/data/output/model/scalaSVMWithSGDModel"
model.save(sc, outputPath)
val sameModel = SVMModel.load(sc, outputPath)