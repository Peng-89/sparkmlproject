package com.za.apps.feature

import org.apache.spark.sql.SparkSession


object VectorSlicer  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("VectorSlicer").master("local[*]").getOrCreate()
    import java.util.Arrays

    import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
    import org.apache.spark.ml.feature.VectorSlicer
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.Row
    import org.apache.spark.sql.types.StructType

    val data = Arrays.asList(
      Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
      Row(Vectors.dense(-2.0, 2.3, 0.0))
    )

    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

    val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

    val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

    //slicer.setIndices(Array(1)).setNames(Array("f3"))
    //slicer.setIndices(Array(1, 2))
    slicer.setNames(Array("f2", "f3"))

    val output = slicer.transform(dataset)
    dataset.show()
    output.show(false)
  }
}
