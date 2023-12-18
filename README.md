# 记录donson实习生经历（设计和参与的算法）
## 文本分类（text_classification）
### input：
标注好的数据
### process：
1.处理数据
2.训练并调参
3.模型推理与评估
### result：已经接入生产。
### best score：
Accuracy in dev dataset: 95.98%。
Macro average : precision: 94.74 | recall: 96.09 | F1 score 95.37
Class name: 0
Evaluation examples in dev dataset: 1411.0(69.1%) | precision: 98.33 | recall: 95.82 | F1 score 97.06
Class name: 1
Evaluation examples in dev dataset: 631.0(30.9%) | precision: 91.15 | recall: 96.35 | F1 score 93.68

## 文本分割（text_segmentation） 纯算法
### input：
文本和关键字
### process：
1.按照标点符号或者第一第二，这些有分类能力的词，将句子分为比较优质的段
2.找到包含关键词的段
3.输出文字段列表
### result：已经接入生产。
### best score：
人工评估500次分割任务，其中包含2k+句子段，分割准确率为97.3%
### todo：
待完成 ①②③④1️⃣2️⃣3️⃣等特殊标志的分割

## 检索算法（search） 纯算法
### input：
文本，prompt（品牌，品类，受众，关键字）
### process：
1.使用whoose实现基于词的匹配baseline
2.利用cos相似度实现基于词义的匹配baseline
3.增加原始数据清洗和处理方法，提高检索成功率
4.利用key，value，query思想，利用百川大模型，将数据库文案总结出key关键词，利用query进行检索
### result：已经接入生产。
### best score：
评测标准选择：查准率和查全率，Precision at K
数据库中有3.7w条数据，抽检1000次query效果
查准率：0.997
查全率：0.953
Precision at K：0.906 （K = 10）

## 营销大模型数据工程(pre_train_dataProcess)
### input:
小红书，微博等爬下来的原始数据
### process:
1.解析数据，parquet格式解析为json格式
2.数据预处理
3.聚类，去除掉语义相似或相同的文案，提高文案多样性（接入公司算法，不能开源）
4.营销文案判断
### result:接入数据流

## chatGLM开发
基于chatGLM开发的程序，具体信息见chatGLM文件夹
### 1.prompt 路由程序
### 2.自由问句提取程序
### 3.意图识别程序
### 4.langchain tool程序


