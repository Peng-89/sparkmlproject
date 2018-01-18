package com.za.apps.classification

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * 回归模型
  */
object LinearRegressionModel {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("LinearRegressionModel").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val records = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\Bike-Sharing-Dataset\\hour.csv")
          .map(_.split(",")).filter(!_(0).equals("instant"))

    //1 of k计算方法中获取类别
    def get_mapping(data:RDD[Array[String]], index:Int)={
      data.map(_(index)).distinct.collect.zipWithIndex.toMap
    }
    //将2-10列所有类别保存在一个数组中，方便后面取
    val mappings = (2 until 10).map(a=>get_mapping(records,a))
    println(mappings.map(_.size).sum)

    def extract_label(record:Array[String])={
      record(record.size-1).toFloat
    }

    //逻辑回归将类别数据转换成二元类型特征
    def extract_features(record:Array[String])={
      var typeFeature:Array[Double]=null
      record.slice(2,10).zipWithIndex.foreach(a=>{
        //1 of K
        val m = mappings(a._2)
        val idx = m(a._1)
        val feature = Array.ofDim[Double](m.size)
        feature(idx)=1.0
        //将所有的特征拼接起来
        if(null == typeFeature ) typeFeature =feature else typeFeature=typeFeature ++ feature
      })
      typeFeature ++ record.slice(10,14).map(_.toDouble)
    }

    //决策树直接使用原始数据，转换成double即可
    def extract_features_dt(record:Array[String])={
      record.slice(2,14).map(_.toDouble)
    }

    val data = records.map(r=>LabeledPoint(extract_label(r),Vectors.dense(extract_features(r))))
    val firest =data.first()
    println(s"Row data:")
    records.first.foreach(a=>print(a+","))
    println()
    println(s"Label:${firest.label}")
    println(s"Linear Model feature vector:${firest.features}")
    println(s"Label Model feature vector length:${firest.features.size}")

    println("决策树特征")
    val data_dt = records.map(r=>LabeledPoint(extract_label(r),Vectors.dense(extract_features_dt(r))))
    val firest_dt =data_dt.first()
    println(s"Label:${firest_dt.label}")
    println(s"Linear Model feature vector:${firest_dt.features}")
    println(s"Label Model feature vector length:${firest_dt.features.size}")

    val numIterations = 10
    val stepSize = 0.1
    //val Array(traindata,testdata) =data.randomSplit(Array(0.7,0.3))
    val lrModel = LinearRegressionWithSGD.train(data,numIterations,stepSize)
    val predicts = data.map(d=>(d.label,lrModel.predict(d.features)))
    println(predicts.take(5).mkString("\n"))

    val valuesAndPreds = data.map { point =>
      val prediction = lrModel.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    val MAE = valuesAndPreds.map{ case(v, p) => math.abs(v - p) }.mean()
    val rmsle = valuesAndPreds.map{ case(v, p) =>  math.pow((math.log(v+1) - math.log(p+1)), 2) }.mean()
    println("Linnear Mean Squared Error = " + MSE)
    println("Linnear Mean Absolute Error = " + MAE)
    println("Linnear Root Mean Squared Log Error = " + rmsle)

    println("决策树模型")
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32
    //val Array(traindatadt,testdatadt) =data_dt.randomSplit(Array(0.7,0.3))
    val dtModel = DecisionTree.trainRegressor(data_dt,categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)
    val predicts_dt = data_dt.map(d=>(d.label,dtModel.predict(d.features)))
    println(predicts_dt.take(5).mkString("\n"))

    val labelsAndPredictions = data_dt.map { point =>
      val prediction = dtModel.predict(point.features)
      (point.label, prediction)
    }
    val MSEdt = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    val MAEdt = labelsAndPredictions.map{ case(v, p) => math.abs(v - p) }.mean()
    val rmsledt = labelsAndPredictions.map{ case(v, p) =>  math.pow((math.log(v+1) - math.log(p+1)), 2) }.mean()
    println("Linnear Mean Squared Error = " + MSEdt)
    println("Linnear Mean Absolute Error = " + MAEdt)
    println("Linnear Root Mean Squared Log Error = " + rmsledt)
    //println("Learned regression tree model:\n" + dtModel.toDebugString)



  }
}
