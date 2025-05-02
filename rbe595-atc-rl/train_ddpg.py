import gymnasium as gym, argparse, envs_patch.register_v1_envs
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SectorCREnv-v1")
    parser.add_argument("--steps", type=int, default=3_000_000)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode=None)
    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(mu=np.zeros(n_actions), sigma=0.3*np.ones(n_actions))

    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=noise,
        tensorboard_log="tb/ddpg",
        verbose=1,
    )
    model.learn(args.steps)
    model.save(f"models/{args.env}-ddpg")