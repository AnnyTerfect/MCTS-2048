# MCTS_2048_AI

## 介绍
用蒙特卡洛树搜索方法实现的会玩2048游戏的智能程序，每1秒执行一个行动，并可视化智能程序玩 2048游戏的过程。

## 目录结构

.

├── Game2048Env.py 2048环境定义

├── agent.py 基于MCTS的智能agent

└── main.py 主程序

## 复现方式

### 安装依赖

程序依赖于numpy, pyqt5, gym, fire这四个库，在使用前请先安装好依赖

```bash
pip install -r requirements.txt
```

或者手动安装以上库。

### 运行主程序

```bash
python main.py \
--seed 0 \
--gamma 0.99 \
--c 100 \
--iter_time 1 \
--d 10 \
--render True
```

其中参数的意义如下

* --seed 设置随机种子
* --gamma MCTS奖赏折扣系数
* --c MCTS探索鼓励系数
* --iter_time simulate步骤限制时间
* --d 探索深度
* --render 是否渲染
