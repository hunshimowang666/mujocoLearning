# Inverted Pendulum RL

这是一个 MuJoCo 小车倒立摆强化学习例子。

## 文件

- `inverted_pendulum_model.xml`：MuJoCo MJCF 模型，小车 + 倒立杆。
- `inverted_pendulum_env.py`：Gymnasium 环境。
- `inverted_pendulum_train.py`：PPO 训练脚本。
- `inverted_pendulum_play.py`：加载训练结果并用 MuJoCo viewer 展示。

## 训练

```bash
cd /home/administrator/mujocoLearning/examples/RLLearning
/home/administrator/mujocoLearning/venv/bin/python inverted_pendulum_train.py
```

训练结果保存在：

```text
examples/RLLearning/logs/
```

## 展示

```bash
cd /home/administrator/mujocoLearning/examples/RLLearning
/home/administrator/mujocoLearning/venv/bin/python inverted_pendulum_play.py
```

## 控制任务

动作空间：

```text
[cart_force]
```

动作范围是 `[-1, 1]`，映射到小车水平力 `[-20 N, 20 N]`。

观测空间：

```text
[cart_x, cart_v, sin(theta), cos(theta), theta_dot, last_action]
```

其中 `theta = 0` 表示杆竖直向上。
