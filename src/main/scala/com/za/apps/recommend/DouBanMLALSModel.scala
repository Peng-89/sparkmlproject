package com.za.apps.recommend

import com.za.apps.recommend.DouBanALSModel.dataPath
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

object DouBanMLALSModel {
  val dataPath = "D:\\workspace\\sparkmlproject\\data\\douban_data\\"

  case class Douban(userId:Int,movieId:Int,rating:Double,userName:String)
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")

    val conf = new SparkConf().setMaster("local[*]").setAppName("DouBanMLALSModel").set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val data = sc.textFile(dataPath + "user_movies.csv").map(_.split(","))
    val userIds = data.map(_(0)).filter(!"".equals(_)).distinct().zipWithIndex().collectAsMap()
    val movieIdMaps  = sc.textFile(dataPath+"hot_movies.csv").map(_.split(",")).map(a=>(a(0).toInt,a(2))).collectAsMap()

    val userIdbrods = sc.broadcast(userIds)

    val ratings = data.map {
      case Array(user, movie, rating) => {
        //如果评分小于0 也就是-1的情况将rating转成3.0
        Douban(userIdbrods.value(user).toInt, movie.toInt,
          if (rating.toDouble>0) rating.toDouble else 3.0,user)
      }
    }
    val datas = sqlContext.createDataFrame(ratings)

    //val Array(trandata,test)=datas.randomSplit(Array(0.8,0.2))

    val trandata =datas

    val als = new ALS()
      .setRank(50)
      .setMaxIter(10)
      .setRegParam(0.0001)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(trandata)

    model.setColdStartStrategy("drop")
    //计算rmse
    val predictions = model.transform(trandata)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    val recommendations =model.recommendForAllUsers(10).where("userId=1238")
    println("userId=1238 看过的电影")
    trandata.where("userId='1238'").rdd.foreach{
      case Row(userId,movieId,rating,userName)=>println(movieIdMaps(movieId.toString.toInt))
    }

    println("userId=1238 推荐的10部电影")
    recommendations.select("recommendations").rdd.foreach(row=>{
      val rs = row.getAs[mutable.WrappedArray[GenericRowWithSchema]]("recommendations")
      rs.foreach {
        case Row(a, b) =>println(movieIdMaps(a.toString.toInt),b)
      }
    })

    model.recommendForAllItems(10).printSchema()

  }
}
