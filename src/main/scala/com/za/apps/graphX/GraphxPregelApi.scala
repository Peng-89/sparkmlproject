package com.za.apps.graphX

import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 2018/1/21.
  */
object GraphxPregelApi {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("GraphxPregelApi").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import org.apache.spark.graphx._
    val myVertices = spark.sparkContext.makeRDD(Array((1l,"Ann"),(2l,"Bill"),(3l,"Charies"),
      (4l,"Diane"),(5l,"Went to gym this morning")))
    val myEdges = spark.sparkContext.makeRDD(Array(Edge(1l,2l,"is-frends-with"),Edge(2l,3l,"is-frends-with")
      ,Edge(3l,4l,"is-frends-with"),Edge(4l,5l,"likes-status"),Edge(3l,5l,"Wrote-status")))
    //通过Graph半生对象来构造Graph对象，其实是Graph.apply方法，需要vertices Rdd 和Edge Rdd
    val myGraph = Graph(myVertices,myEdges)

    /**
      * def pregel[A]
      *     （initialMsg:A,
      *       maxIter:Int = Int.MaxValue
      *       activeDirection=EdgeDirection.Out)
      *       (vprog:(VertexId,VD,A)=>VD,
      *       sendMsg:EdgeTriplet[VD,ED]=>Iterator[(VertexId,A)],
      *       mergeMsg:(A,A)=>A)
      *
      */
    val g = Pregel(myGraph.mapVertices((vid, id) => 0), 0,
      activeDirection = EdgeDirection.Out) ((id: VertexId, vd: Int, a: Int) => math.max(vd, a)
      , (et: EdgeTriplet[Int, String]) => Iterator((et.dstId, et.srcAttr + 1)), (a: Int, b: Int) => math.max(a, b)
       )
    println(g.vertices.collect().mkString("\n"))
  }
}
