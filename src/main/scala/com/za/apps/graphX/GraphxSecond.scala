package com.za.apps.graphX


import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 2018/1/21.
  */
object GraphxSecond {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("GraphxSecond").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import org.apache.spark.graphx._
    val myVertices = spark.sparkContext.makeRDD(Array((1l,"Ann"),(2l,"Bill"),(3l,"Charies"),
      (4l,"Diane"),(5l,"Went to gym this morning")))
    val myEdges = spark.sparkContext.makeRDD(Array(Edge(1l,2l,"is-frends-with"),Edge(2l,3l,"is-frends-with")
            ,Edge(3l,4l,"is-frends-with"),Edge(4l,5l,"likes-status"),Edge(3l,5l,"Wrote-status")))
    //通过Graph半生对象来构造Graph对象，其实是Graph.apply方法，需要vertices Rdd 和Edge Rdd
    val myGraph = Graph(myVertices,myEdges)
    println(myGraph.vertices.collect().mkString("\n"))
    //Edge中存储了源顶点和目标顶点，是因为顶点分布在不同的rdd中，有可能源顶点和目标顶点在不同的机器
    println(myGraph.edges.collect().mkString("\n"))

    //triplets()方法将点和边链接起来，Graph本来是将数据分开存储在点RDD和边RDD内
    //((目标ID，ATTR)，(dstID,dstAttr),Attr)
    val myTriplets =myGraph.triplets
    println(myTriplets.collect().mkString("\n"))

    //mapping操作（graphX很多转换操作都会生成一个新图）
    //mapTriplets 其实也会产生triplets的效果 会在边上产生新的属性，修改之前的属性
    //1.属性包含“is-friends-with” 2.关系的源顶点属性中包含"a",添加（attr，满足条件是否boolean)
    //在边属性上增加布尔类型的属性表示一个条件限制
    println(myGraph.mapTriplets(t=>(t.attr,t.attr.contains("is-frends-with") && t.srcAttr.contains("a"))).triplets.collect().mkString("\n"))
    //mapVertices 添加新的属性
    println(myGraph.mapVertices((a,b)=>(a,b,"1")).triplets.collect().mkString("\n"))

    //aggregateMessages graphx的map reduce 是从边向顶点发送消息，然后聚合消息，遍历所有的边
    /**
      * aggregateMessages 两个参数
      * sengMsg: 发送消息类似map
      *   sendToSrc:将Msg类型的消息发送给源端
      *   sengToDst:将Msg类型的消息发送给源端
      * mergeMsg: 每个顶点所有消息都会被聚集起来 reduce
      */
    //这里统计顶点的出度，因此需要向源顶点发送消息，这里的消息数数字1
    println(myGraph.aggregateMessages[Int](_.sendToSrc(1),_+_).collect().mkString("\n"))

    /**
      *实现迭代 在图中寻找距离最远的根顶点的算法
      *
      * @param g
      * @return
      */
    def propagateEdgeCount(g:Graph[Int,String]):Graph[Int,String]={
      //sendMsg：发送srtAttr+1的数据，merdMsg：合并同一个顶点，取数据最大的
      val verts =g.aggregateMessages[Int](t=>t.sendToDst(t.srcAttr+1),(a,b)=>Math.max(a,b))
      val g2 = Graph(verts,g.edges)
      //如果和上一次迭代的结果，顶点的距离差没有变化则迭代结束
      val check = g2.vertices.join(g.vertices).map(x=>x._2._1-x._2._2).reduce(_+_)
      if(check>0) propagateEdgeCount(g2) else g
    }

    //初始化
    val initialGraph=myGraph.mapVertices((_,_)=>0)

    /**
      * 第1次迭代：(1,0),(2,1),(3,1),(4,1),(5,1){3到5值为1，4到5:1} check:4
      * 第2次迭代：(1,0),(2,1),(3,2),(4,2),(5,2){3到5值为2，4到5:2} 2次迭代-1次迭代的结果check：3
      * 第3次迭代：(1,0),(2,1),(3,2),(4,3),(5,3){3到5值为3，4到5:3} 3次迭代-2次迭代的结果check：2
      * 第4次迭代：(1,0),(2,1),(3,2),(4,3),(5,4){3到5值为3，4到5:4} 3次迭代-2次迭代的结果check：1
      * 第5次迭代：(1,0),(2,1),(3,2),(4,3),(5,4){3到5值为3，4到5:4} 3次迭代-2次迭代的结果check：0
      * 推出
      */
    println(propagateEdgeCount(initialGraph).vertices.collect().mkString("\n"))

    //序列化和反序列化
    //保存数据
    //myGraph.vertices.saveAsObjectFile("file:///F:\\workspace\\sparkmlproject\\data\\myGraphVertices")
    //myGraph.edges.saveAsObjectFile("file:///F:\\workspace\\sparkmlproject\\data\\myGraphEdge")
    //读取文件
//    val myGraph2 = Graph(
//      spark.sparkContext.objectFile[Tuple2[VertexId,String]](""),
//      spark.sparkContext.objectFile[Edge[String]](""))
    //图生成
    //生成网格图
    import org.apache.spark.graphx.util.GraphGenerators
    val gridGraph = GraphGenerators.gridGraph(spark.sparkContext,4,4)
    println(gridGraph.triplets.collect().mkString("\n"))
    //生成星形图
    val startGraph = GraphGenerators.starGraph(spark.sparkContext,8)
    println(startGraph.triplets.collect().mkString("\n"))
    //随机图 单步算法（对数正态算法（出度）） 多步算法（R-Mat算法）
    //正态算法（顶点出度满足（正态分布（高斯钟形曲线）））
    val logNormalGraph = GraphGenerators.logNormalGraph(spark.sparkContext,15)
    println(logNormalGraph.triplets.collect().mkString("\n"))
    println(logNormalGraph.outDegrees.collect().mkString("\t"))
    //R-MAT 代表递归矩阵 用于模拟典型的社交网络,第二个和第三个参数是要求顶点数和独立的边数--顶点数被取值为最近的一个2的幂值
    val rmatGraph = GraphGenerators.rmatGraph(spark.sparkContext,32,60)
  }
}
