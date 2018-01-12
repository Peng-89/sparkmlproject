package com.za.apps.ljemail

import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.mllib.classification.{LogisticRegressionModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._

object ModelLREmailTest {
  val modelpath="D:\\workspace\\sparkmlproject\\data\\model\\modellremail\\"

  case class RawDataRecord(category: String, words: List[String])

  /**
    * String 分词
    *
    * @param sentense
    * @return
    */
  def transform(sentense: String): List[String] = {
    val list = StandardTokenizer.segment(sentense)
    CoreStopWordDictionary.apply(list)
    list.map(x => x.word.replaceAll(" ", "")).toList
  }

  def main(args : Array[String]) {

    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")

    val conf = new SparkConf().setMaster("local[*]").setAppName("ModelLREmailTest")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

//    var datadf = sc.textFile("D:\\workspace\\sparkmlproject\\data\\email_data\\*").map {
//      x =>RawDataRecord(x.substring(0,1),transform(x.substring(2)))
//    }.toDF()
//
//    datadf.repartition(20).write.mode(SaveMode.Overwrite).save("D:\\workspace\\sparkmlproject\\data\\email_data_rep")
    var datadf = sqlContext.read.load("D:\\workspace\\sparkmlproject\\data\\email_data_rep")

    val hashingTF = new HashingTF()
      .setNumFeatures(5000).setInputCol("words").setOutputCol("rawFeatures")
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var featurizedData = hashingTF.transform(datadf)

    var testRdd = sc.textFile("D:\\workspace\\sparkmlproject\\data\\test_data.txt").map {
      x =>RawDataRecord(x.substring(0,1),transform(x.substring(2)))
    }.toDF()

    val featurizedDatat = hashingTF.transform(testRdd)

    var idfModel = idf.fit(featurizedData)

    var rescaledData = idfModel.transform(featurizedDatat)
    var testDataRdd = rescaledData.select($"category", $"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    val nmodel = LogisticRegressionModel.load(sc, modelpath)
    val testpredictionAndLabel2 = testDataRdd.map(p => (nmodel.predict(p.features), p.label))
    testpredictionAndLabel2.collect().foreach(println)
  }
}
