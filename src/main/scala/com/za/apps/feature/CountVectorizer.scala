package com.za.apps.feature

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

object CountVectorizer {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("OneHotEncoder").master("local[*]").getOrCreate()
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a","d","b","b"))
    )).toDF("id", "words")

    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    cvModel.transform(df).show()

    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b","c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvm.transform(df).select("features").foreach(a=>println(a))

  }
}
