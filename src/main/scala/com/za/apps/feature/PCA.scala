package com.za.apps.feature

import org.apache.spark.sql.SparkSession

object PCA {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("PCA").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.PCA
    import org.apache.spark.ml.linalg.Vectors

    //五维的向量
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      //转换成几维的向量，就是降成几维
      .setK(3)
      .fit(df)

    val result = pca.transform(df)
    result.show(false)
  }
}
