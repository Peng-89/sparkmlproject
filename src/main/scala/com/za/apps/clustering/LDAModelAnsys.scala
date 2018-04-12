package com.za.apps.clustering

import java.util.regex.Pattern

import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.JavaConversions._
import scala.collection.mutable

object LDAModelAnsys {

  def delHTMLTag(value:String):String={
    var htmlStr=value
    val regEx_script="<script[^>]*?>[\\s\\S]*?<\\/script>"; //定义script的正则表达式
    val regEx_style="<style[^>]*?>[\\s\\S]*?<\\/style>"; //定义style的正则表达式
    val regEx_html="<[^>]+>"; //定义HTML标签的正则表达式
    val p_script=Pattern.compile(regEx_script,Pattern.CASE_INSENSITIVE);
    val m_script=p_script.matcher(htmlStr);
    htmlStr=m_script.replaceAll(""); //过滤script标签

    val p_style=Pattern.compile(regEx_style,Pattern.CASE_INSENSITIVE);
    val m_style=p_style.matcher(htmlStr);
    htmlStr=m_style.replaceAll(""); //过滤style标签

    val p_html=Pattern.compile(regEx_html,Pattern.CASE_INSENSITIVE);
    val m_html=p_html.matcher(htmlStr);
    htmlStr=m_html.replaceAll(""); //过滤html标签

    // <p>段落替换为换行
    htmlStr = htmlStr.replaceAll("<p .*?>", "\r\n");
    // <br><br/>替换为换行
    htmlStr = htmlStr.replaceAll("<br\\s*/?>", "\r\n");
    // 去掉其它的<>之间的东西
    htmlStr = htmlStr.replaceAll("\\<.*?>", "");

    htmlStr = htmlStr.replaceAll("#","").replaceAll("---","").replaceAll("-","")

    htmlStr.trim
  }
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

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("LDAModelAnsys").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._
   /* val md_data_rdd = spark.sparkContext.textFile("D:\\workspace\\spark-2.1.1\\docs\\*.md").map {
      x =>transform(delHTMLTag(x))
    }.filter(_.size>0)*/
   val md_data_rdd = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\sougou-train\\*").map {
     x =>transform(delHTMLTag(x))
   }.filter(_.size>0)
    val md_data= md_data_rdd.toDF("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(5000).setInputCol("words").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val featurizedData = hashingTF.transform(md_data)
    val idfModel = idf.fit(featurizedData)
    val dataset = idfModel.transform(featurizedData)
    //dataset.show()
    featurizedData.show()

    //用来处理关于hash值和字条的映射 通过indexOf("spark") 获取对应的hash
    import org.apache.spark.mllib.feature.HashingTF
    val hashModel = new HashingTF(5000)
    val hash_value = md_data_rdd.flatMap(a=>a).distinct().map(a=>{
      (hashModel.indexOf(a).toDouble,a)
    }).collectAsMap()

    val lda = new LDA().setK(5).setMaxIter(100).setCheckpointInterval(10)
    val ldmodel = lda.fit(dataset)
    val descTops = ldmodel.describeTopics()
    descTops.rdd.foreach{
      case Row(topic,termIndices,termWeights)=>{
        val ct = termIndices.asInstanceOf[mutable.WrappedArray[Integer]].map(a=>hash_value(a.toString.toDouble)).mkString(",")
        val wi = termWeights.asInstanceOf[mutable.WrappedArray[Double]].mkString(",")
        println(s"${topic},${ct},${wi}")
      }
    }
  }
}
