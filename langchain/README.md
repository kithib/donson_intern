# langchian agent项目实践

## camel

已完成，使用chatglm3代替chatgpt

优点：使用简单的本地模型，快速有效的解决小问题

缺点：受限于chatglm的maxlength（可能使用一些较好的记忆模型）和模型大小，无法解决较难问题，比如股票机器人问题

## Generative Agent

已完成，使用chatglm3代替chatgpt，使用sentence-transformers的embedding代替OpenAIembedding

优缺点与camel相同

相比于camel，Generative Agent加入了记忆模块，利用向量数据库检索记忆，利用反思机制总结记忆，可以让智能体有更长的记忆，简单来说，可以一定程度上解决chatglm的maxlength太小问题。

但是，这种记忆机制并不完善，我的课题则是基于此。

## Gymnasium

已完成，代理-环境交互循环模拟，可用于模拟强化学习

## Simulate a two-player/mutil-player Dungeons & dragons Game

已完成

模拟两人或多人龙与地下城游戏。

设计了对话模拟器用于协调主角与地下城城主之间的对话。
