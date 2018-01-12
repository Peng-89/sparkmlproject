package com.za.apps.recommend

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}

object DouBanALSModel {
  val dataPath = "D:\\workspace\\sparkmlproject\\data\\douban_data\\"

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")

    val conf = new SparkConf().setMaster("local[*]").setAppName("DouBanALSModel").set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val data = sc.textFile(dataPath+"user_movies.csv").map(_.split(","))
//    println(s"用户数量：${data.map(_(0)).filter(!"".equals(_)).count()}," +
//        s"电影数量${data.map(_(1)).filter(!"".equals(_)).count()}," +
//        s"评分数量：${data.map(_(2).toDouble).filter(_>0).count()}")
    //将用户id装成整数
    val userIds = data.map(_(0)).filter(!"".equals(_)).distinct().zipWithIndex().collectAsMap()
    val userIdbrods = sc.broadcast(userIds)
    //-1分的情况转换成3.0
    val ratings = data.map {
      case Array(user, movie, rating) => {
        //如果评分小于0 也就是-1的情况将rating转成3.0
        Rating(userIdbrods.value(user).toInt, movie.toInt,
          if (rating.toDouble>0) rating.toDouble else 3.0)
      }
    }
    //通过计算sme来评估模型的好坏，选择对应的rangk,lambda
    val numIterations=10

    def MSE(model:MatrixFactorizationModel)={
      val userProducts = ratings.map{case Rating(user,product,rating)=>(user,product)}

      val predictions = model.predict(userProducts).map{case Rating(user,product,rating)=>((user,product),rating)}

      val ratingAndPredictions = ratings.map{
        case Rating(user,product,rating)=>((user,product),rating)
      }.join(predictions)

      ratingAndPredictions.map{
        case ((user,product),(actual,predicted))=>math.pow(actual-predicted,2)
      }.reduce(_+_)/ratingAndPredictions.count()
    }

    //选择rank 和 lambda 参数
    def checkParams() {
      for (rank <- Array(10, 50, 70);
           lambda <- Array(1.0, 0.01, 0.0001)) {
        val model = ALS.train(ratings, rank, numIterations, lambda)
        println(s"(rank:$rank, lambda: $lambda, Explicit ) Mean Squared Error = " + MSE(model))
      }
    }

    val model = ALS.train(ratings,50,numIterations,0.0001)
    //136801905,20645098,5
    val userName ="xrzsdan"
    val k = 10
    val userId =userIdbrods.value(userName).toInt
    val recommendations = model.recommendProducts(userId,k)

    //电影名称id对应表
    val movieIdMaps  = sc.textFile(dataPath+"hot_movies.csv").map(_.split(",")).map(a=>(a(0).toInt,a(2))).collectAsMap()
    //用户看过的电影
    println(s"${userName} 看过的电影 ")
    data.filter(_(0).equals(userName)).foreach(a=>println(movieIdMaps(a(1).toInt)))

    //推荐列表
    println("推荐电影top 5 ")
    recommendations.foreach{
      case Rating(user,product,rating)=>{
        println(movieIdMaps(product))
      }
    }

    //物品推荐
    import org.jblas.DoubleMatrix
    val itemId=26253733
    val itemVector = new DoubleMatrix(model.productFeatures.lookup(itemId).head)
    def consineSimilarity(vec1:DoubleMatrix,vec2:DoubleMatrix) ={
      vec1.dot(vec2) /(vec1.norm2() * vec2.norm2())
    }
    println("=================物品推荐")

    val sims = model.productFeatures.map{
      case (id,factor)=>
        val factorVector = new DoubleMatrix(factor)
        val sim = consineSimilarity(factorVector,itemVector)
        (id,sim)
    }

    val sortedSims =sims.top(11)(Ordering.by[(Int,Double),Double]{
      case (id,similarity)=>similarity
    })
    sortedSims.slice(1,11).map{case (id,sim)=>(movieIdMaps(id),sim)}.foreach(println)

  }
}
