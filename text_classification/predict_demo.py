from paddlenlp import Taskflow
# 模型预测
# /home/kit/kit/text_classification/data/checkpoint/export
task_path = '/home/kit/kit/text_classification/data/checkpoint/export'
# task_path = 'checkpoint/export'
cls = Taskflow("text_classification", task_path=task_path, is_static_model=True)
text = "【开学装备】oppo reno10 5G智能手机 8GB+256GB 2399元包邮下单立减 2399元包邮下单立减优惠高考季"
print(cls([text]))
#  [{'predictions': [{'label': '0', 'score': 0.9967874446262931}],
#   'text': '【开学装备】oppo reno10 5G智能手机 8GB+256GB 2399元包邮下单立减 2399元包邮下单立减优惠高考季'}]

# s

