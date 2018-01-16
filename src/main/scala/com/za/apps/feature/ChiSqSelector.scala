package com.za.apps.feature

import org.apache.spark.sql.SparkSession


/**
ChiSqSelector代表卡方特征选择。它适用于带有类别特征的标签数据。ChiSqSelector根据独立卡方检验，然后选取类别标签主要依赖的特征。它类似于选取最有预测能力的特征。它支持三种特征选取方法：
1、numTopFeatures：通过卡方检验选取最具有预测能力的Top(num)个特征；
2、percentile：类似于上一种方法，但是选取一小部分特征而不是固定(num)个特征；
3、fpr:选择P值低于门限值的特征，这样就可以控制false positive rate来进行特征选择；
        默认情况下特征选择方法是numTopFeatures(50)，可以根据setSelectorType()选择特征选取方法。
        示例：假设我们有一个DataFrame含有id,features和clicked三列，其中clicked为需要预测的目标：
id | features              | clicked
---|-----------------------|---------
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0
        如果我们使用ChiSqSelector并设置numTopFeatures为1，根据标签clicked，features中最后一列将会是最有用特征：
id | features              | clicked | selectedFeatures
---|-----------------------|---------|------------------
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0     | [1.0]
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0     | [0.0]
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0     | [0.1]
  */
object ChiSqSelector  {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("ChiSqSelector").master("local[*]").getOrCreate()
    import spark.implicits._
    import org.apache.spark.ml.feature.ChiSqSelector
    import org.apache.spark.ml.linalg.Vectors

    val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )

    val df = spark.createDataFrame(data).toDF("id", "features", "clicked")

    val selector = new ChiSqSelector()
        .setNumTopFeatures(1)
      //.setPercentile(0.1)
        //.setFpr(0.1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(df).transform(df)

    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()
  }
}
