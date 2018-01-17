package com.za.apps.classification

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.sql.SparkSession

/**
  * 模型调优
  */
object ClassifcationModelParams {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\服务器管理\\平台软件\\hadoop-2.6.4\\hadoop-2.6.4")
    val spark = SparkSession.builder().appName("ClassifcationModelParams").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val numIterations = 10
    val maxTresDepth = 5
    val data = spark.sparkContext.textFile("D:\\workspace\\sparkmlproject\\data\\train.tsv")
    val records = data.map(_.split("\t")).filter(!_(3).equals("\"alchemy_category\""))
    //构建向量
    var rddLabledPoint= records.map{ r=>
      val trimmed = r.map(_.replaceAll("\"",""))
      val label =trimmed(r.size-1).toInt
      //将？ 转成0
      val features = Vectors.dense(trimmed.slice(4,r.size-1).map(d=>if (d =="?") 0.0 else d.toDouble))
      LabeledPoint(label,features)
    }
    //finalData.show()
    rddLabledPoint.cache()
    //println(finalData.count())

    //对特征标准化
    //统计特征值每列特性，通过RowMatrix来处理
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    val vectors = rddLabledPoint.map(_.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //均值
    println(matrixSummary.mean)
    println(matrixSummary.min)
    println(matrixSummary.max)
    //方差
    println(matrixSummary.variance)
    //非0项的数目
    println(matrixSummary.numNonzeros)

    /**
      * 对每个特征进行标准化使得每个特征值是0均值和单位标准差
      * （x-u）/sqrt(variance)
      * x:特征值 u：每个特征值均值 sqrt(variance)：标准差以进行缩放
      * spark 中提供了StandardScaler 特征值处理标准化的方法
      */
    import org.apache.spark.mllib.feature.StandardScaler
    //withMean 表示是否减去均值，withStd是否应用标准差缩放
    val scaler = new StandardScaler(withMean = true,withStd = true).fit(vectors)
    val scaladData = rddLabledPoint.map(lp=>LabeledPoint(lp.label,scaler.transform(lp.features)))
    println("==========使用spark标准化之后的数据==========")
    println(rddLabledPoint.first().features)
    println(scaladData.first().features)

    rddLabledPoint=scaladData

    val categories = records.map(_(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    println(categories)

    // 1 of k
    val dataCategories = records.map{r=>
      val trimmed = r.map(_.replaceAll("\"",""))
      val label =trimmed(r.size-1).toInt
      val categoryIdx = categories(r(3))
      val categoryFetures = Array.ofDim[Double](numCategories)
      categoryFetures(categoryIdx)=1.0
      val otherFeatures =  trimmed.slice(4,r.size-1).map(d=>if (d =="?") 0.0 else d.toDouble)
      val features = categoryFetures ++ otherFeatures
      LabeledPoint(label,Vectors.dense(features))
    }
    println(dataCategories.first())

    val scalerCats = new StandardScaler(withMean = true,withStd = true).fit(dataCategories.map(_.features))
    rddLabledPoint = dataCategories.map(lp=>LabeledPoint(lp.label,scalerCats.transform(lp.features)))

    //添加了类别以后
    println("==================添加了类别特征值=================================")
    println(rddLabledPoint.first().features)

    //逻辑回归线性模型
    val lrModel = LogisticRegressionWithSGD.train(rddLabledPoint,numIterations)
    import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
    //val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(numIterations).run(rddLabledPoint)


    //SVM
    val svmModel =SVMWithSGD.train(rddLabledPoint,numIterations)

    //贝叶斯
    //requireNonnegativeValues：取每一个样本数据的特征值，以向量形式存储，特征植必需为非负数。
    val nbdata = rddLabledPoint.map(a=>LabeledPoint(a.label,Vectors.dense(a.features.toArray.map(a=>if (a<0) 0.0 else a))))
    val nbModel = NaiveBayes.train(nbdata)
    //决策树
    val dtModel = DecisionTree.train(rddLabledPoint,Algo.Classification,Entropy,maxTresDepth)

    val dataPoint = rddLabledPoint.first()
    println(s"true Label :${dataPoint.label}")
    val prediction =lrModel.predict(dataPoint.features)
    println(s"prediction:${prediction}")

    //整体预测
    val predictions = lrModel.predict(rddLabledPoint.map(_.features))
    predictions.take(5).foreach(println)

    //评估分类模型的性能
    /**
      * 预测正确率和错误率
      * 准确率和召回率
      * 准确率-召回率曲线下面的面积
      * ROC曲线
      * ROC曲线下的面积
      * F—Meatsure
      */
    //1.预测正确率和错误率
    /**
      * 预测正确率:正确率等于训练样本中被正确分类的数目除以总样本数
      * 错误率：训练样本中被错误分类的样本数目除以总样本数
      */
    val lrTotalCorrect = rddLabledPoint.map{
      point=>if(lrModel.predict(point.features)==point.label) 1 else 0
    }.sum()
    val lrAccouracy = lrTotalCorrect/rddLabledPoint.count()
    println(s"lrAccouracy:${lrAccouracy}")

    val svmAccouracy = rddLabledPoint.map{
      point=>if(svmModel.predict(point.features)==point.label) 1 else 0
    }.sum()/rddLabledPoint.count()
    println(s"svmAccouracy:${svmAccouracy}")

    val nbAccouracy = nbdata.map{
      point=>if(nbModel.predict(point.features)==point.label) 1 else 0
    }.sum()/nbdata.count()
    println(s"nbAccouracy:${nbAccouracy}")

    //决策树给出的是预测值，因此需要指定阀值来判断
    val dtAccouracy = rddLabledPoint.map{ point=>
      val score = dtModel.predict(point.features)
      if((if(score>0.5) 1 else 0)==point.label) 1 else 0
    }.sum()/rddLabledPoint.count()
    println(s"dtAccouracy:${dtAccouracy}")

    //2.准确率和召回率
    /**
      * 通常用于评价结果的质量，而召回率用来评价结果完整性
      * 准确率:真阳性的数目除以真阳性和假阳性的总数，其中真阳性是指被正确预测的类别为1的样本，假阳性是错误预测为类别1的样本，
      *         如果每个被分类器预测为类别1的样本确实属于类别1，那准确率达到100%
      *召回率：真阳性的数目除以真阳性和假阴性的和，其中假阴性是类别为1却被预测为0的样本 真阴性：类别为0被预测为0的样本
      */
    val lrZhenyang = rddLabledPoint.map{ point=>
      val pred = lrModel.predict(point.features)
      //真阳性
      var zy=0
      //假阳性
      var jy=0
      //假阴性
      var jjy=0
      if(pred==1 && point.label==1) zy=1
      if(pred==1 && point.label==0) jy=1
      if(pred==0 && point.label==1) jjy=1
      (zy,jy,jjy)
    }
    val lrzql = lrZhenyang.map(_._1).sum()/(lrZhenyang.map(_._1).sum()+lrZhenyang.map(_._2).sum())
    val lrzhl = lrZhenyang.map(_._1).sum()/(lrZhenyang.map(_._1).sum()+lrZhenyang.map(_._3).sum())
    println(s"逻辑回顾准确率：${lrzql},召回率："+lrzhl)

    //ROC曲线 和ROC曲线下的面积(AUC)
    /**
      * ROC曲线是对分类器真阳性率-假阳性率的图形化解释
      * 真阳性率（TPR）：真阳性的样本数除以真阳性和假阴性的样本数之和 （类似召回率） 通常称为敏感度
      * 假阳性率（FPR）：假阳性的样本数除以假阳性和真阴性的样本数之和 真阴性：类别为0被预测为0的样本
      */
      //MLIb内置方法来计算PR（准确率-召回率曲线）和ROC曲线 BinaryClassificationMetrics
    val metrics =Seq(lrModel,svmModel).map{model=>
        val scoreAndLabels = rddLabledPoint.map(point=>(model.predict(point.features),point.label))
        val metres = new BinaryClassificationMetrics(scoreAndLabels)
        (model.getClass.getSimpleName,metres.areaUnderPR(),metres.areaUnderROC())
      }

    val nbMetercs = Seq(nbModel).map{model=>
      val scoreAndLabels = nbdata.map{point=>
        val score =model.predict(point.features)
        (if (score>0.5) 1.0 else 0.0 ,point.label)}
      val metres = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName,metres.areaUnderPR(),metres.areaUnderROC())
    }

    val dtMetrics = Seq(dtModel).map{model=>
      val scoreAndLabels = nbdata.map{point=>
        val score =model.predict(point.features)
        (if (score>0.5) 1.0 else 0.0 ,point.label)}
      val metres = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName,metres.areaUnderPR(),metres.areaUnderROC())
    }

    val allMetrics = metrics ++ nbMetercs ++ dtMetrics

    allMetrics.foreach{
      case (m,pr,roc)=>
        println(f"$m,Area under PR:${pr*100}%2.4f%%,Area under ROC:${roc*100}%2.4f%%")
    }

  }
}
