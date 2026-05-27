#!/bin/bash
# setup.sh — 一键安装所有依赖
# 用法: bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "============================================"
echo "  MuJoCo RL 仿真环境一键安装脚本"
echo "============================================"

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] 创建虚拟环境..."
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

source "$VENV_DIR/bin/activate"
echo "  虚拟环境: $VENV_DIR"

# 升级 pip
echo "[2/4] 升级 pip..."
pip install --upgrade pip -i $MIRROR -q

# 安装核心依赖
echo "[3/4] 安装 MuJoCo + Gymnasium..."
pip install mujoco "gymnasium[mujoco]" -i $MIRROR --timeout 120

echo "[4/4] 安装 stable-baselines3 + torch..."
pip install "stable-baselines3[extra]" torch torchvision \
    -i $MIRROR --timeout 300

echo ""
echo "============================================"
echo "  安装完成！验证命令:"
echo "  source venv/bin/activate"
echo "  python examples/04_quick_demo.py"
echo "============================================"
