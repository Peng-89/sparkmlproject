package com.za.apps.graphX

import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 2018/1/21.
  */
object GraphxFirst {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("GraphxFirst").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import org.apache.spark.graphx._
    //通过GraphLoader.edgeListFile 加载标准的边列表的数据
    val graph = GraphLoader.edgeListFile(spark.sparkContext,"F:\\workspace\\sparkmlproject\\data\\Cit-HepTh.txt")
    //inDegress 可以计算每个顶点的入度（顶点ID，入度），reduce来计算出最大入度的顶点
    val maxDegress = graph.inDegrees.reduce((a,b)=>if(a._2>b._2) a else b)
    println(maxDegress)

    //vertices 是顶点数据集合（ID,属性） edgeListFile方法默认设置属性为1
    println(graph.vertices.take(10).mkString("\n"))
    //edges是边数据集合 （源顶点ID，目标顶点ID，属性） edgeListFile方法默认设置属性为1
    println(graph.edges.take(10).mkString("\n"))

    //通过pageRank计算v 将原来的图转成一个新图
    val v =graph.pageRank(0.001).vertices
    println(v.take(10).mkString("\n"))
    //pageRank值最高的顶点
    println(v.reduce((a,b)=>if(a._2>b._2) a else b))
  }
}
