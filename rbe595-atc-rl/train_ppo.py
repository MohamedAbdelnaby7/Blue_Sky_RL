import gymnasium as gym, argparse, envs_patch.register_v1_envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from torch import multiprocessing as mp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SectorCREnv-v1")
    parser.add_argument("--steps", type=int, default=3_000_000)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode=None)
    eval_env = gym.make(args.env, render_mode=None)

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="tb/ppo")
    model.learn(
        args.steps,
        callback=EvalCallback(eval_env, best_model_save_path="models/ppo"),
    )
    model.save(f"models/{args.env}-ppo")