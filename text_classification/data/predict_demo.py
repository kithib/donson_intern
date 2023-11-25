from paddlenlp import Taskflow
# 模型预测
# /home/kit/kit/text_classification/data/checkpoint/export
task_path = '/home/kit/kit/text_classification/data/checkpoint_old/export'
# task_path = 'checkpoint/export'
cls = Taskflow("text_classification", task_path=task_path, is_static_model=True)
text = "药店一盒就36的[傻眼]咱这可是2盒啊！！！  【拍2件💥27.4】江中 利活乳酸菌素*128片  之前64片就要18.9了，咱这是2倍的量！好肠胃是乳酸菌片，调理出来的👌🏻这个亲测好使！补充益生菌+调节肠道均衡，吃饭不消化/肠胃不好/便秘，家中常备！"
print(cls([text]))
#  [{'predictions': [{'label': '0', 'score': 0.9967874446262931}],
#   'text': '【开学装备】oppo reno10 5G智能手机 8GB+256GB 2399元包邮下单立减 2399元包邮下单立减优惠高考季'}]
#[{'predictions': [{'label': '0', 'score': 0.9999937296579788}], 'text': '药店一盒就36的[傻眼]咱这可是2盒啊！！！  【拍2件💥27.4】江中 利活乳酸菌素*128片  之前64片就要18.9了，咱这是2倍的量！好肠胃是乳酸菌片，调理出来的👌🏻这个亲测好使！补充益生菌+调节肠道均衡，吃饭不消化/肠胃不好/便秘，家中常备！'}]

# s
