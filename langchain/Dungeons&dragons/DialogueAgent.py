from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import sys
sys.path.append('/home/kit/kit/donson_intern/langchain')
from ChatGLM3 import ChatGLM3
llm = ChatGLM3()


#DialogueAgent 类是 llm 模型的一个简单包装器,通过将消息连接为字符串来存储dialogue_agent中的消息历史记录。
class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatGLM3,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
 
    def reset(self):
        self.message_history = ["Here is the conversation so far."]


    # 将历史消息输入到llm中，输出返回信息
    def send(self) -> str:
        prompt = self.system_message.content + '/n'
        prompt += "".join(self.message_history + [self.prefix])
        message = self.model(prompt)
        return message
    

    #将信息存入message history，形式为 {message} spoken by {name} 
    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f"{name}: {message}")