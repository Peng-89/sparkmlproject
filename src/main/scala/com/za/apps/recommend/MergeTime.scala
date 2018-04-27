package com.za.apps.recommend

import java.text.SimpleDateFormat
import java.util
import java.util.{Calendar, Date}

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer


case class mtime(userid: String, locationid: String, begintime: String, split: Int,endtime:String)

object MergeTime {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("MergeTime").set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    sc.setLogLevel("ERROR")
    val data = sc.textFile("file:///D:\\workspace\\sparkmlproject\\data\\time.txt")
      .map(_.split(","))
      .map(a => {
        val calender = Calendar.getInstance()
        calender.setTime(sdf.parse(a(2).toString))
        calender.add(Calendar.MINUTE,a(3).toInt)
        mtime(a(0).toString, a(1).toString, a(2).toString, a(3).toInt,sdf.format(calender.getTime))
      })

    data.map(a => (a.userid + "_" + a.locationid, a))
      .groupByKey()
      .map(a => {
        val values = a._2.toList.sortBy(_.begintime)
        val length = values.length
        var index = 0
        val dataList = new ListBuffer[mtime]
        var first =values(index)
        var tmp = first
        while(index<length){
            if(index == length-1){
              first = values(index )
              tmp = values(index )
              dataList +=tmp
            }else {
              if (tmp.endtime.equals(values(index + 1).begintime)) {
                tmp = mtime(first.userid, first.locationid, first.begintime, first.split + values(index + 1).split, values(index + 1).endtime)
              } else {
                dataList +=tmp
                first = values(index + 1)
                tmp = values(index + 1)
              }
            }
            index=index+1
        }
        (a._1,dataList)
      }).foreach(println)
  }
}
