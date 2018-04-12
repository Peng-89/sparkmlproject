package com.za.apps.recommend

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


object TestClass {
  System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
  val conf = new SparkConf().setMaster("local[*]").setAppName("Test").set("spark.executor.memory", "4g")
  @transient
  val sc = new SparkContext(conf)

  val hiveContext= new SQLContext(sc)

  def main(args: Array[String]): Unit = {

    sc.setLogLevel("ERROR")


    val data = sc.parallelize(Array(1,2,3,4),1)

    //val ba = sc.broadcast(hiveContext)
    data.foreachPartition(x=>{
      hiveContext.sql("show tables").show()
    })

  }
}
