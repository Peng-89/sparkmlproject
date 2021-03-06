package com.za.apps.feature

import org.apache.spark.sql.SparkSession


/**
分箱（分段处理）：将连续数值转换为离散类别
        比如特征是年龄，是一个连续数值，需要将其转换为离散类别(未成年人、青年人、中年人、老年人），就要用到Bucketizer了。
        分类的标准是自己定义的，在Spark中为split参数,定义如下：
        double[] splits = {0, 18, 35,50， Double.PositiveInfinity}
        将数值年龄分为四类0-18，18-35，35-50，55+四个段。
     如果左右边界拿不准，就设置为，Double.NegativeInfinity， Double.PositiveInfinity，不会有错的。
  */
object Bucketizer  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("Bucketizer").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.Bucketizer

    val splits = Array(0, 18, 35, 50,Double.PositiveInfinity)

    val data = Array(12,19,20,36,53)
    val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)

    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)

    println(s"Bucketizer output with ${bucketizer.getSplits.length-1} buckets")
    bucketedData.show()
  }
}
