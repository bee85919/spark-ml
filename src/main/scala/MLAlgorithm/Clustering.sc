import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg. Vectors

// Load and parse the data
val sparkHome = System.getenv("SPARK_HOME")
val data = sc.textFile(s"${sparkHome}/data/mllib/kmeans_data.txt")
val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans

val numClusters = 2

val numlterations = 20

val clusters = KMeans.train(parsedData, numClusters, numlterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors =" + WSSSE)

// Save and load model
val outputPath = s"${sparkHome}/data/output/model/KMeansModel"
clusters.save(sc, outputPath)
val sameModel = KMeansModel.load(sc, outputPath)