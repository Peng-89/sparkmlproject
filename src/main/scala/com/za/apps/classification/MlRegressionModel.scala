package com.za.apps.classification

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

/**
  * 回归模型ml
  */
object MlRegressionModel {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("MlRegressionModel").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val records = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\Bike-Sharing-Dataset\\hour.csv")
          .map(_.split(",")).filter(!_(0).equals("instant"))

    //1 of k计算方法中获取类别
    def get_mapping(data:RDD[Array[String]], index:Int)={
      data.map(_(index)).distinct.collect.zipWithIndex.toMap
    }
    //将2-10列所有类别保存在一个数组中，方便后面取
    val mappings = (2 until 10).map(a=>get_mapping(records,a))
    //println(mappings.map(_.size).sum)

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

    import spark.implicits._

    val data = records.map(r=>(math.log(extract_label(r)),Vectors.dense(extract_features(r)))).toDF("label","features")
    val numIterations = 100

    val params= Seq(0.0,0.01,0.1,1.0,100,100.0,1000.0)
    val rmse= for(ni<-params) yield {
      val lrModel = new LinearRegression()
        .setMaxIter(numIterations)
        .setRegParam(ni)
        .setElasticNetParam(0.8).fit(data)

//      val test = data.limit(1)
//      test.show
//      lrModel.transform(test).show

      val trainingSummary = lrModel.summary

      //println(s"iterations:${ni},stepsize:${st},RME:${trainingSummary.meanSquaredError}")
      trainingSummary.rootMeanSquaredError
      //println(s"iterations:${ni},stepsize:${st},r2: ${trainingSummary.r2}")
    }
    println(params)
    println(rmse)

  }
}
