# 2048-NN-Agent
作为课程作业而开发本项目，基于tensorflow的神经网络，在ExpectiMax代理监督下学习、训练网络，训练完成后可作为智能代理使用

# 代码结构
* [`game2048/`](game2048/): 主要代码包.
    * [`game.py`](game2048/game.py): 2048核心类 `Game`.
    * [`myagents.py`](game2048/myagents.py): `NNAgent` 代理类.
    * [`displays.py`](game2048/displays.py): 显示游戏状态的类.
    * [`expectimax/`](game2048/expectimax): 基于ExpectiMax 的代理[这里](https://github.com/nneonneo/2048-ai).
* [`model/`](model/): 预训练的神经网络模型.
* [`evaluate.py`](evaluate.py): 用于评估自定义的NNAgent.
* [`generate_fingerprint.py`](generate_fingerprint.py):用于生成基于自定义代理的指纹.

# Requirements
* 参考 requirements.txt

# LICENSE
The code is under Apache-2.0 License.
