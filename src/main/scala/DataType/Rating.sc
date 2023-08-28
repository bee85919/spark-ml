import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Load and parse the data

val sparkHome = System.getenv("SPARK_HOME")
val data = sc.textFile(s"${sparkHome}/data/mllib/als/test.data")

val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)

})