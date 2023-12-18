import torch
import time
import math
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/tds/ChatGLM3/chatglm3-6b",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/tds/ChatGLM3/chatglm3-6b",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


model, tokenizer = init_model()
model.eval()

def gen_prompt(user_question):
    prompt = f'''
        对于用户的输入：“{user_question}”，请评估以下准则
                    1、用户输入为包含特定日期或时间范围，例如询问“今天、明天、后天、最新”
                    2、用户输入为询问某一具体事件
                    3、用户输入为询问公司、公司、影视剧、电影等相关介绍
                    4、用户输入为询问实时股市信息和财经数据：股票价格、市场趋势、交易数据等
                    5、用户输入为询问天气预报等
                    6、用户输入为询问实时体育比赛分数和结果
                    7、用户输入为询问特定区域的交通状况和旅行建议
                    8、用户输入为询问最新的法律和政策变化
                    9、用户输入为询问名人和公众人物的最新消息
                    10、用输入题为询问最新科技产品和硬件发布信息
                    11、用户输入为询问特定商品和服务的最新价格信息：如电子产品、航班票价、酒店预订
                    12、用户输入为询问公共服务和设施的运营信息
                    13、用户输入为你无法回答，不知道答案的
                    14、用户输入为询问语句，非操作指令
                    如果用户输入符合以上任一准则，请输出“是”，表示需要进行联网搜索。如果用户输入没有涉及上述准则，请输出“否”。记住，你只能输出“是”或者“否”，请严格遵守输出规则。
                        '''
    return prompt
# 判断是否需要联网搜索
def intent_judgment(user_question):
    # 规则判断
    if '营销' or '文案' or '帮我' or '写' or '生成' in user_question:
        return False
    else:
    # 调用大模型判断
        # 构建参数
        prompt = gen_prompt(user_question)
        # 调用大模型
        answer = model(prompt)
        # 如果答案是 "否'，则表示大模型回答，否则 联网搜索
        if answer == "否":
            return False
        else:
            return True

