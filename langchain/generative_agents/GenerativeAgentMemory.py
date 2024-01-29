import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now
import sys
sys.path.append('/home/kit/kit/donson_intern/langchain/langchain/libs/core/langchain_core/language_models')
from base import BaseLanguageModel
sys.path.append('/home/kit/kit/langchain')
from ChatGLM3 import ChatGLM3

logger = logging.getLogger(__name__)

class GenerativeAgentMemory(BaseMemory):
    # memory for the generative agents

    # The core language model.
    llm: ChatGLM3

    # The retriever to fetch related memories
    memory_retriever: TimeWeightedVectorStoreRetriever

    # 是否记录中间环节
    verbose: bool = False

    # 当aggregate_importance超过reflection_threshold ，智能体停止思考
    reflection_threshold: Optional[float] = None

    # The current plan of the agent
    current_plan: List[str] = []
    
    # A weight of 0.15 makes this less important than it would be otherwise, relative to salience and time
    importance_weight: float = 0.15

    # 为记忆的重要性分配多少权重
    aggregate_importance: float = 0.0  # : :meta private:

    #######  跟踪近期记忆的重要性的总和，当达到reflection_threshold时触发反应  ########

    #最大tokens限制
    max_tokens_limit: int = 1200  # : :meta private:


    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"


    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    reflecting: bool = False

    # 建立chain
    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
    
    #@staticmethod是一个装饰器，用于将一个方法标记为静态方法。
    #静态方法不需要访问或修改类实例的状态，也不需要访问或修改任何类的属性。
    #它们基本上是普通的函数，只是被放在类里面，并且可以通过类来调用。
    #静态方法可以通过类名直接调用，而不需要创建类的实例。
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        # 将换行分割的字符串，解析为字符串列表
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # 去掉空line
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines] #去掉每行开头的数字和点号

    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        # 返回关于最近观察的三个最显著的高层次问题
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:] #检索最近last_k个observations
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]  #加入创建时间，并用换行符连接
        )
        result = self.chain(prompt).run(observations=observation_str) #将observation_str加入template中，将整个问题输入chain中
        return self._parse_list(result) #返回解析后的字符串列表

    def _get_insights_on_topic(
        self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:
        # 根据相关记忆，对反思的主题产生见解
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories) # enumerate(related_memories)可以输出related_memories的序号和内容
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        ) #将related_statements放入template中，输入chain
        # TODO: 利用 insight (because of 1, 5, 3) ，解析出见解和记忆的联系
        #match = re.search(r'\(because of (\d+(, \d+)*)\)', result) 
        #if match:
        #    reasonOfinsight = match.group(1)
        return self._parse_list(result)

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        # 对近期的发现进行反思，并且生成见解，整合上面两个内部调用函数
        if self.verbose:
            logger.info("Character is reflecting") # 输出提示
        new_insights = [] # 建立存见解存储列表
        topics = self._get_topics_of_reflection() #获取最近的3个反思
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now) #对于每个反思获取见解
            for insight in insights:
                self.add_memory(insight, now=now) #将见解加入记忆中
            new_insights.extend(insights) # 将不同反思的见解合并
        return new_insights

    def _score_memory_importance(self, memory_content: str) -> float:
        # 对给定记忆的绝对重要性进行评分
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip() # 将需要评分的内容和评分规则输入chain
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score) #搜索一个或多个数字
        if match: #搜索到了
            return (float(match.group(1)) / 10) * self.importance_weight # 换算分数为0-1范围内，并且与权重相乘
        else:
            return 0.0 # 否则 返回0.0

    def _score_memories_importance(self, memory_content: str) -> List[float]:
        # 对给定多条记忆进行评分
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colans (;)"
            + "\Memories: {memory_content}"
            + "\nRating: "
        )
        scores = self.chain(prompt).run(memory_content=memory_content).strip()

        if self.verbose:
            logger.info(f"Importance scores: {scores}")

        # Split into list of strings and convert to floats
        scores_list = [float(x) for x in scores.split(";")]

        return scores_list

    def add_memories(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        # 将发现和记忆加入智能体的记忆中
        importance_scores = self._score_memories_importance(memory_content)
        # 计算并更新aggregate_importance,达到一定重要性阈值开始反思
        self.aggregate_importance += max(importance_scores)
        memory_list = memory_content.split(";")
        documents = []

        for i in range(len(memory_list)):
            documents.append( # 将memory转为Document格式存储，同时存储记忆内容和记忆重要性
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        result = self.memory_retriever.add_documents(documents, current_time=now) #将增加的记忆加入记忆搜索器中

        # 在智能体处理了一定数量的记忆（通过aggregate_importance来衡量）后，
        # 当aggregate_importance超过阈值后，开始反思最近的事件
        # 将更多的合成记忆添加到代理的记忆流中。
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting # 确定智能体没有在反思
        ):
            self.reflecting = True  # 智能体在反思
            self.pause_to_reflect(now=now)
            # 反思后重置aggregate_importance
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def add_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        # 增加单个记忆
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result

    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        # 获取相关记忆
        #如果不是现在，或者未输入，则模拟现在
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- ")) #返回 前缀（-） + 创建时间 + 内容
        return "\n".join([f"{mem}" for mem in content])# 利用换行符连接成字符串

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p") #加入创建时间
        return f"{prefix}[{created_time}] {memory.page_content.strip()}" #返回 前缀 + 创建时间 + 内容

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        #利用分号连接成字符串
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories]) 

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit: # 如果使用的token大于等于最大的max_tokens_limit，则不继续增加
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content) # 计算consumed_tokens
            if consumed_tokens < self.max_tokens_limit: # 如果使用的token小于最大的max_tokens_limit，则继续增加
                result.append(doc)
        return self.format_memories_simple(result)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key) # 从input得到queries
        now = inputs.get(self.now_key)# 读取时间
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=now) #利用query获取记忆
            ]
            return { #返回字典
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key) # 得到最近记忆value
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit( # 将记忆确定在maxlength以内
                    most_recent_memories_token
                )
            }
        return {}# 返回空

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # 保存记忆
        # TODO: fix the save memory key
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO