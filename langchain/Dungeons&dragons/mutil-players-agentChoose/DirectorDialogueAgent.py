import random
import tenacity
from typing import List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import sys
sys.path.append('/home/kit/kit/donson_intern/langchain/Dungeons&dragons')
from DialogueAgent import DialogueAgent

sys.path.append('/home/kit/kit/donson_intern/langchain')
from ChatGLM3 import ChatGLM3
llm = ChatGLM3()


#DirectorDialogueAgent 是一个特权代理，可以选择下一个要说话的代理。该代理负责
#1.通过选择代理使对话更顺畅
#2.终止对话
#首先， 要引导对话， DirectorDialogueAgent 需要在单个消息中
#（1）反思先前的谈话
#（2）选择下一个代理
#（3）提示下一个代理发言。
#虽然在同一调用中提示 LLM 执行所有三个步骤是可能的，但这需要编写自定义代码来解析输出的消息，以提取选择下一个代理的信息。这不太可靠，因为 LLM 可以用不同的方式表达它选择下一个代理的方式。
#相反，我们可以将步骤（1-3）明确地分成三个单独的 LLM 调用。
#首先，我们将要求 DirectorDialogueAgent 反思到目前为止的对话并生成一个响应。
#然后我们提示 DirectorDialogueAgent 输出下一个代理的索引，这很容易解析。
#最后，我们将选择的下一个代理的名称传回 DirectorDialogueAgent 以要求其提示下一个代理发言。
#第二，仅仅提示"DirectorDialogueAgent"何时终止对话往往会导致立即终止对话。 
#为了解决这个问题，我们随机采样Bernoulli变量来决定对话是否应该终止。根据这个变量的值，我们将注入自定义的提示，告诉"DirectorDialogueAgent"是继续对话还是终止对话。

class IntegerOutputParser(RegexParser):# 大模型响应形式规定
    def get_format_instructions(self) -> str:
        return 'Your response should be an integer delimited by angled brackets, like this: <int>.' 
 
class DirectorDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        model: ChatGLM3,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.next_speaker = ''
        self.stop = False
        self.stopping_probability = stopping_probability # 停止可能性
        # 中断规则，以总结性信息或者感谢结束对话
        self.termination_clause = 'Finish the conversation by stating a concluding message and thanking everyone.' 
        # 继续规则，添加你自己的想法继续对话
        self.continuation_clause = 'Do not end the conversation. Keep the conversation going by adding your own ideas.'

        # 1. 对前一位发言者做出回应的prompt
        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}
Follow up with an insightful comment.
{{termination_clause}}
{self.prefix}
 """)

        # 2. 决定下一个发言者
        self.choice_parser = IntegerOutputParser(
            regex=r'<(\d+)>',  # <(\d+)> 会匹配一个以 < 开头，后跟一个或多个数字，并以 > 结尾的字符串。
            output_keys=['choice'],  # 若匹配到，字典key为choice，value为数字
            default_output_key='choice')   # 若未匹配到，字典key为choice，value为None
        self.choose_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "speaker_names"],
            template=f"""{{message_history}}
Given the above conversation, select the next speaker by choosing index next to their name: 
{{speaker_names}}
{self.choice_parser.get_format_instructions()}
Do nothing else.
 """)

        # 3. 提示下一个发言者发言
        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}
The next speaker is {{next_speaker}}. 
Prompt the next speaker to speak with an insightful question.
{self.prefix}
 """)
    
    def _generate_response(self):
        # 如果 self.stop = True,将会在prompt中注入termination_clause
        sample = random.uniform(0,1) # 生成一个在闭区间 [0, 1]（包括0和1）内的随机浮点数。
        self.stop = sample < self.stopping_probability # 随机浮点数低于stopping_probability，则停止
        print(f'- Stop? {self.stop}')
        response_prompt = self.response_prompt_template.format(
            message_history=''.join(self.message_history),
            termination_clause=self.termination_clause if self.stop else ''
        )
        prompt = self.system_message.content + '\n' + response_prompt
        self.response = self.model(prompt)
        return self.response
 
#tenacity库是Python中用于增强函数或方法的健壮性的一个库，特别是在面对临时故障（如网络中断、数据库连接问题等）时。


#@tenacity.retry(...): 这是一个装饰器，用于包装一个函数或方法，使其具有重试的逻辑。
#总的来说，这段代码的主要目的是确保当某个函数抛出ValueError异常时，它会尝试重新执行该函数最多2次，并且每次重试之间没有等待时间。如果所有重试都失败，它会调用一个回调函数并返回
    @tenacity.retry(stop=tenacity.stop_after_attempt(2),# 这意味着如果函数连续失败2次，重试将停止。
                    wait=tenacity.wait_none(),  # 每次重试之间没有等待时间。
                    retry=tenacity.retry_if_exception_type(ValueError),# 如果函数抛出ValueError异常，则触发重试
                    before_sleep=lambda retry_state: print(f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),# 在每次重试之前，都会打印出关于异常的信息。
                    retry_error_callback=lambda retry_state: 0) # 当所有重试都失败时，这个回调函数会被调用,返回0
 
    def _choose_next_speaker(self) -> str:        
        speaker_names = ''.join([f'{idx}: {name}' for idx, name in enumerate(self.speakers)]) # 将所有参与者的id和姓名组合
        choice_prompt = self.choose_next_speaker_prompt_template.format(
            message_history=''.join(self.message_history + [self.prefix] + [self.response]),
            speaker_names=speaker_names
        )
        prompt = self.system_message.content + '\n' + choice_prompt
        choice_string = self.model(prompt)
        choice = int(self.choice_parser.parse(choice_string)['choice'])
        return choice

    def select_next_speaker(self):
        return self.chosen_speaker_id
 
    def send(self) -> str:

        # 1. 生成并保存先前发言者的发言
        self.response = self._generate_response()
        if self.stop:
            message = self.response

        else:
            # 2. 决定下一个发言者
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print(f'- Next speaker: {self.next_speaker}')

            # 3. 提示下一个发言者发言
            next_prompt = self.prompt_next_speaker_prompt_template.format(
                message_history="".join(self.message_history + [self.prefix] + [self.response]),
                next_speaker=self.next_speaker
            )
            prompt = self.system_message.content + '\n' + next_prompt
            message = self.model(prompt)
            message = ' '.join([self.response, message])
        return message
 