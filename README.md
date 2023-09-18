# text_classification
Marketing copywriting identification
### 运行环境：
- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.5.1
- scikit-learn >= 1.0.2
**安装PaddlePaddle：**
 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。
**安装PaddleNLP：**
```shell
pip install --upgrade paddlenlp
```
**安装sklearn：**
```shell
pip install scikit-learn
```
### 加载模型并预测：
模型文件为/checkpoint，使用时将/checkpoint文件放入项目目录中,使用taskflow进行模型预测，请保证paddlenlp版本大于2.5.1:
```
from paddlenlp import Taskflow
# 模型预测
cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
text = "【开学装备】oppo reno10 5G智能手机 8GB+256GB 2399元包邮下单立减 2399元包邮下单立减优惠高考季"
cls([text])
# [{'predictions': [{'label': '0', 'score': 0.9967874446262931}],
  'text': '【开学装备】oppo reno10 5G智能手机 8GB+256GB 2399元包邮下单立减 2399元包邮下单立减优惠高考季'}]
```
#### 可配置参数说明
* `task_path`：自定义任务路径，默认为None。
* `is_static_model`：task_path中是否为静态图模型参数，默认为False。
* `max_length`：最长输入长度，包括所有标签的长度，默认为512。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `id2label`：标签映射字典，如果`task_path`中包含id2label.json或加载动态图参数无需定义。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。
