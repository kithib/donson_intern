import pandas as pd
from paddlenlp import Taskflow
df_data = pd.read_excel('predict_data.xlsx')
df_data.drop_duplicates(subset=['content_clean'],keep='first',inplace=True)
for i in range(int(df_data.shape[0])):
    df_data.iloc[i,0] = str(df_data.iloc[i,0]).replace('\t','')
    df_data.iloc[i,0] = str(df_data.iloc[i,0]).replace('\n','')
    df_data.iloc[i,0] = str(df_data.iloc[i,0]).replace(' ','')
df_data = df_data.reset_index(drop = True)
df_data = df_data.assign(Bpredict_label = None, Bscore = None)
cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
for i in range(int(df_data.shape[0])):
    df_data.iloc[i,3] = cls(str(df_data.iloc[i,0]))[0]["predictions"][0]["label"]
    df_data.iloc[i,4] = cls(str(df_data.iloc[i,0]))[0]["predictions"][0]["score"]
    print(str(i) + " , " + str(df_data.shape[0]))
df_data.to_excel('predict_qu_data.xlsx', index=False, engine='openpyxl')