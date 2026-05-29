"""
04_quick_demo.py
================
快速验证脚本 — 无需训练，直接用随机策略跑通所有环境
用于确认安装是否正常

用法:
  python 04_quick_demo.py           # 验证所有环境
  python 04_quick_demo.py --env ant # 只验证某个环境
"""

import argparse
import sys
import traceback
import gymnasium as gym

ENVIRONMENTS = {
    "humanoid":  ("Humanoid-v5",      "人形机器人"),
    "ant":       ("Ant-v5",           "四足机械狗"),
    "cheetah":   ("HalfCheetah-v5",   "猎豹/高速小车"),
    "hopper":    ("Hopper-v5",        "单脚跳机器人"),
    "walker":    ("Walker2d-v5",      "双脚行走机器人"),
    "swimmer":   ("Swimmer-v5",       "游泳机器人"),
    "reacher":   ("Reacher-v5",       "二连杆机械臂"),
    "pusher":    ("Pusher-v5",        "推物机械臂"),
}


def test_env(env_id, name, n_steps=200):
    print(f"\n{'='*55}")
    print(f"  测试: {name} ({env_id})")
    print(f"{'='*55}")
    try:
        env = gym.make(env_id)
        obs, info = env.reset(seed=42)

        print(f"  观测空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        print(f"  初始观测 shape: {obs.shape}")

        total_reward = 0.0
        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        print(f"  [OK] {n_steps} 步随机策略, 累计奖励: {total_reward:.2f}")
        return True

    except Exception as e:
        print(f"  [FAIL] 错误: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, choices=list(ENVIRONMENTS.keys()),
                        help="只测试特定环境")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  MuJoCo 环境快速验证")
    print("="*55)

    import mujoco
    print(f"\n  MuJoCo 版本: {mujoco.__version__}")
    import gymnasium
    print(f"  Gymnasium 版本: {gymnasium.__version__}")

    envs_to_test = (
        {args.env: ENVIRONMENTS[args.env]} if args.env
        else ENVIRONMENTS
    )

    results = {}
    for key, (env_id, name) in envs_to_test.items():
        results[name] = test_env(env_id, name, args.steps)

    print("\n" + "="*55)
    print("  测试结果汇总")
    print("="*55)
    ok_count = sum(results.values())
    for name, ok in results.items():
        status = "✓ 正常" if ok else "✗ 失败"
        print(f"  {status}  {name}")
    print(f"\n  通过: {ok_count}/{len(results)}")

    if ok_count == len(results):
        print("\n  [全部通过] 环境已就绪，可以开始训练！")
        print("  训练命令:")
        print("    python 01_humanoid_train.py    # 人形机器人")
        print("    python 02_ant_dog_train.py     # 机械狗(Ant)")
        print("    python 03_car_cheetah_train.py # 猎豹小车")
    else:
        print("\n  [部分失败] 请检查上方错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
