package com.za.apps.classification

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionMLModel{
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("LinearRegressionMLModel").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    // $example on$
    // Load and parse the data
    val data = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\lpsa.data")
    import spark.implicits._
    val training = data.map { line =>
      val parts = line.split(',')
      (parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.toDF("label","features")

    val lrModel = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8).fit(training)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RME:${trainingSummary.meanSquaredError}")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}
