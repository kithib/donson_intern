# 营销大模型预训练数据流
## 1.数据预处理
obsfile2json.ipynb
data_process.ipynb
2.聚类
### 聚类主函数：
python main_nlp.py  --config_path=/home/kit/clustering4server_simple/data/sentiment/config_sentiment.json #注意一下显卡是否被占用
### 聚类结果转excel：
python show_result_excel.py
数据清洗：DataFiltering.ipynb
3.营销文案判断
text_classification.ipynb

