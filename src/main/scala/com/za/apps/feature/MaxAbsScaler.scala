package com.za.apps.feature

import org.apache.spark.sql.SparkSession

/**
MaxAbsScaler转换Vector行的数据集，通过分割每个特征的最大绝对值来重新缩放每个特征到范围[-1,1]。
它不会移动/居中数据，因此不会破坏任何稀疏性。
MaxAbsScaler计算数据集的汇总统计并生成一个MaxAbsScalerModel。该模型可以将每个特征分别转换为范围[-1,1]
  */
object MaxAbsScaler  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("MaxAbsScaler").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.MaxAbsScaler
    import org.apache.spark.ml.linalg.Vectors

    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")

    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MaxAbsScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [-1, 1]
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.select("features", "scaledFeatures").show()
  }
}
