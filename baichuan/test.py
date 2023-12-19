from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import torch

def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/tds/Baichuan2/Baichuan2-13B-Chat",
        torch_dtype=torch.float16,
        device_map="cuda:2",
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



if __name__ == "__main__":
    model, tokenizer = init_model()
    prompt = """
你是一位精通营销知识和文案策划的小红书爆款写作专家，您的任务通过学习样例文案的语言风格，创作出满足以下要求的小红书文风内容，你不允许回答“好的，当然可以”等描述性词语，也不允许回答“标题，正文”等格式性内容，请勿直接照抄样例文案，你只需要将创作好的内容返回给用户即可，
               要求：  
                   主题：推荐卡尔文·克莱恩Calvin Klein品牌的内衣
                   品牌卖点： 夏款时尚配色、缤纷色彩、花样繁多、超细纤维面料、舒适透气、金典款式、简约大方、高级感、CK、CK内衣、CK男士内裤、CK男士平角裤、男士礼物、男士内裤推荐、超级舒适的男士内裤、夏季男士内裤、夏季冰丝内裤、CK代言人田柾国、CK代言人JENNIE、BLACKPINK、BTS防弹少年团、CK段宜恩、CK主打
                   受众群体：
                   字数要求：51左右
                   营销目的：使用户看了有分享给朋友的想法或者购买欲望
                   写作要求：使用惊叹号、表情符号等增加文章活力，运用悬念来引发读者好奇心
               样例文案：
               ```
               夏款时尚配色，缤纷色彩🍃，花样繁多 CK主打💯超细纤维面料，舒适透气✅ 金典款式，简约大方，高级感拉🈵
               ```
"""
    response ,history = model.chat(tokenizer,prompt,[])
    print(response)