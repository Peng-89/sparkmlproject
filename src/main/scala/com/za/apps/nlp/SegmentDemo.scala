package com.za.apps.nlp

import com.hankcs.hanlp.dictionary.CustomDictionary
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.tokenizer.StandardTokenizer
import scala.collection.JavaConversions._

object SegmentDemo {
  def main(args: Array[String]) {
    val sentense = "1:Received: from hp-5e1fe6310264 ([218.79.188.136])\tby spam-gw.ccert.edu.cn (MIMEDefang) with ESMTP id j7CAoGvt023247\tfor <lu@ccert.edu.cn>; Sun, 14 Aug 2005 09:59:04 +0800 (CST)Message-ID: <200508121850.j7CAoGvt023247@spam-gw.ccert.edu.cn>From: \"yan\"<(8月27-28,上海)培训课程>Reply-To: yan@vip.163.com\"<b4a7r0h0@vip.163.com>To: lu@ccert.edu.cnSubject: =?gb2312?B?t8eyxs7xvq3A7bXEssbO8bncwO0to6jJs8XMxKPE4qOp?=Date: Tue, 30 Aug 2005 10:08:15 +0800MIME-Version: 1.0Content-type: multipart/related;    type=\"multipart/alternative\";    boundary=\"----=_NextPart_000_004A_2531AAAC.6F950005\"X-Priority: 3X-MSMail-Priority: NormalX-Mailer: Microsoft Outlook Express 6.00.2800.1158X-MimeOLE: Produced By Microsoft MimeOLE V6.00.2800.1441                      非财务纠淼牟莆窆芾�-（沙盘模拟）                     ------如何运用财务岳硖岣吖芾砑ㄐ�                                     　                               [课 程 背 景]   　   每一位管理和技术人员都清楚地懂得，单纯从技术角度衡量为合算的方案，也许   却是一个财务陷阱，表面赢利而暗地里亏损，使经   营者无法接受。如何将技术手段与财务运作相结合，使每位管理和技术人员都从   老板的角度进行思考,有效地规避财务陷阱，实现管理决策与居\uE023勘甑囊恢滦裕�   本课程通过沙盘模拟和案例分析，使企业各级管理和技术人员掌握财务管理知识   ，利用财务信息改进管理决策，实现管理效益最大化。通过学习本课程，您将：   ★ 对会计与财务管理有基本了解,提高日常管理活动的财务可行性;   ★ 掌握业绩评价的依据和方法，评估居\uE031导�,实施科学的业绩考核;   ★ 掌握合乎财务栽虻墓芾砭霾叻椒�,与老板的思维同步；   ★ 通过分析关键业绩指标，形成战略规划与全面预算；   ★ 突出企业管理的重心，形成管理的系统性。                               [课 程 大 纲]   　   一、财务工作内容及作用   1、财务会计的基本岳�   2、财务专家的思维模式   3、财务工作的基本内容   4、管理者如何利用财务进行管理和决策   二、如何阅读和分析财务报表   1、会计报表的构成   2、损益表的阅读与分析   3、资产负债表的阅读与分析   4、资金流量和现金流量表的阅读与分析   5、会计报表之间的关系   6、如何从会计报表读懂企业居\uE036纯�   ◆案例分析：读报表，判断企业业绩水平   三、如何运用财务手段进行成本控制   1、产品成本的概念和构成   2、ＣＶＰ（本Ａ浚利）分析与运用   3、标准成本制度在成本控制中的作用   4、如何运用目标成本法控制产品成本，保证利润水平   5、如何运用ABC作业成本法进行管理分析，实施精细成本管理   6、如何针对沉没成本和机会成本进行正确决策   7、如何改善采购、生产等环节的运作以改良企业的整体财务状况   ◆综合案例分析   四、如何运用财务岳硌≡窨尚械墓芾碛爰际醴桨�   1、管理和技术方案的可行性分析   2、新产品开发中的财务可行性分析   3、产品增产/减产时的财务可行性分析   4、生产设备改造与更新的决策分析   5、投资项目的现金流分析   6、投资项目评价方法（净现值法分析，资金时间价值分析）   ◆综合案例演练   五、公司费用及其控制   1、公司费用的构成   2、控制费用的方法   3、影响费用诸因素的分析   4、如何针对成本中心进行费用控制   5、如何针对利润中心进行业绩考核   6、如何针对投资中心进行业绩评价   六、如何利用财务数据分析并改善居\uE01Bㄐ�   1、公司财务分析的核心思路   2、关键财务指标解析   3、 盈利能力分析：资产回报率、股东权益回报率、资产流动速率   4、风险指数分析：流动比率、负债/权益比率、营运偿债能力   5、财务报表综合解读：综合运用财务信息透视公司运作水平   ◆案例分析：某上市公司的财务状况分析与评价   七、企业运营管理沙盘模拟经   营模拟：体验式教学，每个小组5-6人模拟公司的生产、销售和财务分析过程，�   г痹谀Ｄ舛钥怪邢嗷ゼしⅰ⑻逖榫霾哂刖�   营企业的乐趣，同时讲师与学员共同分享解决问题的模型与工具，使学员\"身同�   迨�\"，完成从know-what 向know-why转化。                               [导 师 简 介]   　   Mr                     Wang，管理工程硕士、高级炯檬Γ\uE0D4\uE30B手耙蹬嘌凳π�   会认证职业培训师，历任跨国公司生产负责人、工业工程经   理、管理会计分析师、营运总监等高级管理职务多年，同时还担任<价值工程>杂   志审稿人、辽宁省营口市商业银行独立董   事等职务，对企业管理有较深入的研究。王老师主要从事IE技术应用、成本控制   、管理会计决策等课程的讲授，先后为IBM、TDK、松下、可口可乐、康师傅、汇   源果汁   、雪津啤酒、吉百利食品、冠捷电子、INTEX明达塑胶、正新橡胶、美国ITT集团   、广上科技、美的空调,中兴通讯、京信通信、联想电脑，应用材料(中国)公司�   \uE76B\uE0DA松�   -金山石化、中国化工进出口公司、正大集团大福饲料、厦华集团、灿坤股份、N   EC东金电子、太原   钢铁集团、PHILIPS、深圳开发科技、大冷王运输制冷、三洋华强、TCL等知名企   业提供项目辅导或专题培训。王老师授课狙榉岣唬\uE0D2绺裼哪\uE0F5缎场⒙呒�   清晰、过程互动，案例生动、深受学员喜爱。                              [授课时间/地点]   　   8月27-28 （周六、日） 上海                               [课 程 费 用]                                    　                1980元/人（包含培训费用、午餐、证书、资料).                        优惠：三人以上参加，赠送一名额                               [联 系 我 们]                                    　                               联系人：桂先生              电话：021-58219359    传真：021-58219402                                  E-mail:  gsb158@163.com "
//    CustomDictionary.add("日  期")
//    CustomDictionary.add("版  号")
//    CustomDictionary.add("标  题")
//    CustomDictionary.add("作  者")
//    CustomDictionary.add("正  文")
    val list = StandardTokenizer.segment(sentense)
    CoreStopWordDictionary.apply(list)
    println(list.map(x => x.word.replaceAll(" ","")).mkString(","))

    println("1:233".substring(2))
  }
}
