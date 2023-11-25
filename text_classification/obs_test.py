from obs import *
import io
import pandas as pd

from paddlenlp import Taskflow

ak = "SFDDLWTCP3BEQUP0JPIK"
sk = "HGqOMwVce3sa0U0fjidtuKLPeKinxzfLFSuBUvYh"
server = 'https://obs.cn-south-1.myhuaweicloud.com'
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk,
                      server=server)
path = 'prod/mlm/dwd/dwd_huawei_copywriting_ss/'

resp = obsClient.listObjects('donson-mip-data-warehouse',
                             prefix=path + "part",
                             max_keys=100)
# 模型预测
# /home/kit/kit/text_classification/data_0727/checkpoint/export
task_path = '/home/kit/kit/text_classification/data_0727/checkpoint/export'
# task_path = 'checkpoint/export'
cls = Taskflow("text_classification", task_path=task_path,
               is_static_model=True)

for key in resp.body.contents:
  print(key['key'])

  # 二进制数据流
  resp = obsClient.getObject('donson-mip-data-warehouse',
                             key['key'],
                             loadStreamInMemory=True)
  compressed_stream = io.BytesIO(resp.body.buffer)

  df = pd.read_parquet(compressed_stream)

  labels = []
  scores = []
  for i in cls(list(df['content_clean'])):
    labels.append(i['predictions'][0]['label'])
    scores.append(i['predictions'][0]['score'])

  df['label'] = labels
  df['score'] = scores

  # 现在你可以访问表格数据

  parquet_data = df.to_parquet()

  binary_data = io.BytesIO(parquet_data)

  # print(binary_data.read())

  resp = obsClient.putContent('donson-mip-data-warehouse',
                              'prod/mlm/dws/dws_huawei_marketing_tag_ss/' + key['key'].replace(path, ''), content=binary_data)

