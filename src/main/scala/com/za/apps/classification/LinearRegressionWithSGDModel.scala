package com.za.apps.classification

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql.SparkSession

object LinearRegressionWithSGDModel{
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("LinearRegressionWithSGDModel").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    // $example on$
    // Load and parse the data
    val data = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\lpsa.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    // Building the model
    val numIterations = 10
    val stepSize = 0.3
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println(valuesAndPreds.take(5).mkString("\n"))
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    val MAE = valuesAndPreds.map{ case(v, p) => math.abs(v - p) }.mean()
    val rmsle = valuesAndPreds.map{ case(v, p) =>  math.pow((math.log(v+1) - math.log(p+1)), 2) }.mean()
    println("Linnear Mean Squared Error = " + MSE)
    println("Linnear Mean Absolute Error = " + MAE)
    println("Linnear Root Mean Squared Log Error = " + rmsle)
  }
}
