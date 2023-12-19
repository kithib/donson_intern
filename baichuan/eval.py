import torch
import time
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

#加载模型
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer
model, tokenizer = init_model()

def init_model_withParamater():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        do_sample=True, 
        top_p=8, 
        temperature=1.2,
        repetition_penalty = 1.1,
        num_beams = 2,
        early_stopping = True,
        no_repeat_ngram_size=3
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer
model_withParamater, tokenizer_withParamater = init_model_withParamater()

#加载数据

def duplicate_rows(df, n):  
    # 创建一个新的空DataFrame  
    new_df = pd.DataFrame()  
      
    # 遍历df的每一行  
    for index, row in df.iterrows(): 
        tem_df = df.iloc[index:index+1]  
        for j in range(n):
            new_df = pd.concat([new_df,tem_df]) 
    return new_df



df = pd.read_csv("/home/kit/kit/baichuan/example_data.csv",sep=',')
df = df.drop(columns='Unnamed: 0', axis=1) 
df = df.astype(str)
df.replace('nan', ' ', inplace=True)   
df = df.assign(generate_sentence = None,generate_sentence_withParamater = None)
#df = duplicate_rows(df,20)


prompt = """
你是一位精通营销知识和文案策划的小红书爆款写作专家，您的任务通过学习样例文案的语言风格，创作出满足以下要求的小红书文风内容，你不允许回答“好的，当然可以”等描述性词语，也不允许回答“标题，正文”等格式性内容，请勿直接照抄样例文案，你只需要将创作好的内容返回给用户即可，
               要求：  
                   主题：推荐卡尔文·克莱恩Calvin Klein品牌的内衣
                   品牌卖点： 夏款时尚配色、缤纷色彩、花样繁多、超细纤维面料、舒适透气、金典款式、简约大方、高级感、CK、CK内衣、CK男士内裤、CK男士平角裤、男士礼物、男士内裤推荐、超级舒适的男士内裤、夏季男士内裤、夏季冰丝内裤、CK代言人田柾国、CK代言人JENNIE、BLACKPINK、BTS防弹少年团、CK段宜恩、CK主打
                   受众群体： 
                   平台文风：小红书
                   使用场景： 
                   推荐角度： 
                   文案风格： 
                   故事性： 
                   字数要求： 
                   营销目的： 
                   写作要求：使用惊叹号、表情符号等增加文章活力，运用悬念来引发读者好奇心
                   其他要求： 
               样例文案：
               ```
               夏款时尚配色，缤纷色彩🍃，花样繁多 CK主打💯超细纤维面料，舒适透气✅ 金典款式，简约大方，高级感拉🈵
               ```
"""
for i in range(int(len(df))):
    s = prompt.replace(prompt[729:780],df.iloc[i,12])
    df.iloc[i,10] = str(df.iloc[i,10])
    s = s[:181] + df.iloc[i,0] + s[200:203] + df.iloc[i,1] + s[205:231] +  df.iloc[i,2] + s[383:408] +  df.iloc[i,4] + s[409:434] +  df.iloc[i,3] + s[437:462] +  df.iloc[i,5] + s[463:488]+  df.iloc[i,6] + s[489:514]+  df.iloc[i,7] + s[514:539]+  df.iloc[i,8] + s[539:565]+  df.iloc[i,9] + s[565:591]+  df.iloc[i,10] + s[591:672] + df.iloc[i,11] + s[672:]  
    messages = []
    messages.append({"role": "user", "content": s})
    response = model.chat(tokenizer, messages)
    response_withParamater = model_withParamater.chat(tokenizer_withParamater, messages)
    print(i,s,response,response_withParamater)
    df.iloc[i,15] = response
    df.iloc[i,16] = response_withParamater

output_file = 'output.csv'  
df.to_csv(output_file, index=False)

