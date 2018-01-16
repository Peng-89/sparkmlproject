package com.za.apps.feature

import org.apache.spark.sql.SparkSession


object StandardScaler {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("StandardScaler").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.StandardScaler

    val dataFrame = spark.read.format("libsvm").load("D:\\workspace\\sparkmlproject\\data\\sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    println(scalerModel.mean)
    scaledData.show()
  }
}
