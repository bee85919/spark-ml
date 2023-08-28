import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Load and parse the data
val sparkHome = System.getenv("SPARK_HOME")
val data = sc.textFile(s"${sparkHome}/data/mllib/als/test.data")

val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

// Build the recommendation model using ALS

val rank =10

val numlterations= 10

val lambda =0.01 // regularization parameter

val model = ALS. train(ratings, rank, numlterations, lambda)

// Evaluate the model on rating data
val usersProducts = ratings. map { case Rating(user, product, rate) => (user, product) }
val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) => ((user, product), rate) } // rating 을 리턴
val ratesAndPreds = ratings.map { case Rating(user, product, rate) => ((user, product), rate) }.join(predictions)
val MSE =ratesAndPreds.map { case ((user, product), (r1,r2)) => // r1 = actual, r2 = prediction
  val err=(r1-r2)
  err * err
}.mean()
println("Mean Squared Error =" + MSE)

// Save and load model
val outputPath = s"${sparkHome}/data/output/model/myCollaborativeFilter"
model.save(sc, outputPath)
val sameModel = MatrixFactorizationModel.load(sc, outputPath)