package com.za.apps.feature

import org.apache.spark.sql.SparkSession

/**
VectorAssembler是一个将给定的列列表组合成单个向量列的变换器。将原始特征和由不同特征变换器生成的特征组合成一个特征向量，以便训练ML逻辑回归和决策树等模型是有用的。
VectorAssembler接受以下输入列类型：所有数字类型，布尔类型和矢量类型。在每一行中，输入列的值将按照指定的顺序连接成一个向量。
  */
object VectorAssembler  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("VectorAssembler").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors

    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")

    val output = assembler.transform(dataset)
    println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.show(false)
  }
}
