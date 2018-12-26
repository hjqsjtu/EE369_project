# 2048-api
EE369 课程作业：2048游戏神经网络代理 NNAgent

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): 基于tflearn神经网络的智能代理.
    * [`displays.py`](game2048/displays.py): 课程项目提供的原始显示类.
    * [`expectimax/`](game2048/expectimax): ExpectiMax，课程提供的训练监督用代理.
* ['model/'](model/):
	* [`SJTU_EE369_HJQ.model`](SJTU_EE369_HJQ.model):已经训练的模型
* [`evaluate.py`](evaluate.py): 智能代理评估器.

#  Requirements
* 参见requirements.txt文件

# 测试代理
python evaluate.py>my.log

# 印记生成
python # 测试代理
python generate_fingerprint.py

# LICENSE
The code is under Apache-2.0 License.
