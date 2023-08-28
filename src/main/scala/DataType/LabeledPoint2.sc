import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util. MLUtils
import org.apache.spark.rdd.RDD

val sparkHome = System.getenv("SPARK_HOME")
val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, s"${sparkHome}/data/mllib/sample_libsvm_data.txt")