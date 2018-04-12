package com.za.apps.nlp

import java.io.{File, FileOutputStream}

import com.hankcs.hanlp.dictionary.CustomDictionary
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.tokenizer.StandardTokenizer

import scala.collection.JavaConversions._

object SegmentDemo {
  def main(args: Array[String]) {
    val sentense = "重庆市重庆市江北区海尔路"
    CustomDictionary.add("高利贷")
    CustomDictionary.add("作流水")
    CustomDictionary.add("代办流水")
    (0 to 100).foreach(a=>{
      CustomDictionary.add("作  者"+a)
    })
    val list = StandardTokenizer.segment(sentense)
    CoreStopWordDictionary.apply(list)
    println(list.map(x => x.word.replaceAll(" ","")).mkString(","))

  }
}
