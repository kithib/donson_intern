from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from DialogueAgent import DialogueAgent


#DialogueSimulator 类 负责智能体讲话列表。
#在每个步骤中，它执行以下操作 :
#1. 选择下一位发言人，方法（循环遍历代理）也可以是其他的
#2. 呼叫下一位发言者发送消息
#3. 将消息广播给所有其他代理
#4. 更新计数器。

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
 
    def reset(self):
        for agent in self.agents:
            agent.reset()
 
 
    # 初始化谈话，初始化消息形式 {message} from {name}
    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)


        # increment time
        self._step += 1
 
 
 
    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
 
        # 2. next speaker sends message
        message = speaker.send()
 
        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
 
        # 4. increment time
        self._step += 1
        
        return speaker.name, message
 
 
 