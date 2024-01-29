import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import sys
sys.path.append('/home/kit/kit/donson_intern/langchain/langchain/libs/core/langchain_core')
from language_models.base import BaseLanguageModel
from GenerativeAgentMemory import GenerativeAgentMemory
from pydantic_v1 import BaseModel, Field
sys.path.append('/home/kit/kit/langchain')
from ChatGLM3 import ChatGLM3

class GenerativeAgent(BaseModel):
    # 智能体是具有记忆和先天特征的角色

    # 角色的姓名
    name: str
   
    # 角色的年龄，可选
    age: Optional[int] = None

    # 角色的性格特征，注意：赋予角色的永久特征。
    traits: str = "N/A"

    # 你不希望改变的角色特征。
    status: str

    # 角色的记忆，与记忆相关性、新近度和重要性的有关的记忆对象
    memory: GenerativeAgentMemory
    
    # 底层语言模型
    llm: ChatGLM3

    # 是否记录详细信息
    verbose: bool = False

    #通过对角色记忆的反思而生成的状态自我总结。
    summary: str = ""  #: :meta private:

    # 反思总结更新时间
    summary_refresh_seconds: int = 3600  #: :meta private:

    # 记录上一次反思总结更新的时间
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:

    # 智能体采取的计划中的事件摘要。
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    

    class Config:
        # Configuration for this pydantic object.

        arbitrary_types_allowed = True

    # LLM-related methods
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        # 将换行分割的字符串，解析为字符串列表
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines] #删除开头的，空白字符，数字，.

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        #建立chain
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        # 得到观察实体
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        #得到观察实体的行为
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        # 总结与观察最相关的记忆。
        prompt = PromptTemplate.from_template(
            """
Please answer in English and never say other language.
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
            """
        )
        entity_name = self._get_entity_from_observation(observation) #得到命名实体
        entity_action = self._get_entity_action(observation, entity_name) #得到命名实体动作
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip() #relevant_memories由queries检索记忆得到

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        # 对给定的观察或对话行为做出反应
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict( # 将参数压缩为字典
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens( #计算prompt token数
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens # 参数字典增加一项，存储consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip() # 将参数字典解压，输入chain中

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip() #删除self.name开头的部分，并且去除空字符

    def generate_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        # 对给定的观察做出反应，可能包括动作和交谈
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now  #  call_to_action_template 为后缀
        )
        result = full_result.strip().split("\n")[0] #得到result
        # AAA
        self.memory.save_context(
            {},# inputs为空
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },# outputs 
        )
        if "REACT:" in result: # 行为
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result: # 对话
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        # 对给定的观察做出反应，对话行为
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now  #  call_to_action_template 为后缀
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result: # end the conversation
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context( #保存在记忆里
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result: # 继续交谈
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def _compute_agent_summary(self) -> str:
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish "
            + "use English."
            + "\n\nSummary: "
        )
        # 让智能体思考他们的核心特征
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        # 返回一个描述性的智能体总结
        current_time = datetime.now() if now is None else now # 获取现在时间
        since_refresh = (current_time - self.last_refreshed).seconds # 获取距离上次更新的间隔
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds # 需要更新
            or force_refresh # 强制更新
        ):
            self.summary = self._compute_agent_summary() #获得更新信息
            self.last_refreshed = current_time #更新上次更新时间
        age = self.age if self.age is not None else "N/A" # 获得年龄
        return ( # 返回总结信息
            f"Name: {self.name} (age: {age})"
            + f"\nInnate traits: {self.traits}"
            + f"\n{self.summary}"
        )

    def get_full_header(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        # 返回一个智能体的全部信息，包括状态，总结和当前时间
        now = datetime.now() if now is None else now 
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )