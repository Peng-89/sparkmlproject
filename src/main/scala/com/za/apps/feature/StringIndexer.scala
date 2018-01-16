package com.za.apps.feature

import org.apache.spark.sql.SparkSession

/**
  * StringIndexer将字符串标签编码为标签指标。指标取值范围为[0,numLabels]，按照标签出现频率排序，
  * 所以出现最频繁的标签其指标为0。如果输入列为数值型，我们先将之映射到字符串然后再对字符串的值进行指标。
  * 如果下游的管道节点需要使用字符串－指标标签，则必须将输入和钻还为字符串－指标列名。
  */
object StringIndexer {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("StringIndexer").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.StringIndexer

    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val indexModel =indexer.fit(df)

    val indexed = indexModel.transform(df)
    indexed.show()

    val df2 = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"), (6, "d"))
    ).toDF("id", "category")

    /**
      * 如果训练好StringIndexer的模型以后，如果测试数据中出现新的类别，则会报错
      * setHandleInvalid("keep") 来设置，如果设置keep，则会将数据添加到训练集中，出现新的Index
      * 如果设置成skip则会忽略新的数据
      */
    //indexModel.setHandleInvalid("keep")
    indexModel.setHandleInvalid("skip")
    indexModel.transform(df2).show()

  }
}
