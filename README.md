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
待完成 ①②③④1️⃣2️⃣3️⃣ 分割