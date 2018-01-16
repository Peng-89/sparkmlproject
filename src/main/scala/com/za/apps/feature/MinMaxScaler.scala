package com.za.apps.feature

import org.apache.spark.sql.SparkSession

/**
  * MinMaxScaler转换Vector行数据集，将每个特征重新缩放到特定范围（通常为[0，1]）。它需要参数：

min：默认为0.0。转换后的下界，所有功能共享。
max：默认为1.0。变换后的上界，被所有的特征共享。
MinMaxScaler计算数据集的汇总统计并生成一个MinMaxScalerModel。然后模型可以单独转换每个特征，使其在给定的范围内。
  */
object MinMaxScaler  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("MinMaxScaler").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.MinMaxScaler
    import org.apache.spark.ml.linalg.Vectors

    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(dataFrame)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
    scaledData.select("features", "scaledFeatures").show()
  }
}
