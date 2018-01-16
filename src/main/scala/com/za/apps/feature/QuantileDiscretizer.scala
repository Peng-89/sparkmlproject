package com.za.apps.feature

import org.apache.spark.sql.SparkSession

/**
分位树为数离散化，和Bucketizer（分箱处理）一样也是：将连续数值特征转换为离散类别特征。实际上Class QuantileDiscretizer extends （继承自） Class（Bucketizer）。

参数1：不同的是这里不再自己定义splits（分类标准），而是定义分几箱(段）就可以了。QuantileDiscretizer自己调用函数计算分位数，并完成离散化。
-参数2： 另外一个参数是精度，如果设置为0，则计算最精确的分位数，这是一个高时间代价的操作。
另外上下边界将设置为正负无穷，覆盖所有实数范围。
  */
object QuantileDiscretizer  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("QuantileDiscretizer").master("local[*]").getOrCreate()
    import org.apache.spark.ml.feature.QuantileDiscretizer

    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    val df = spark.createDataFrame(data).toDF("id", "hour")

    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)
      .setRelativeError(0)

    val result = discretizer.fit(df).transform(df)
    result.show()
  }
}
