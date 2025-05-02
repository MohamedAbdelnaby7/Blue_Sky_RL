import gymnasium as gym, envs_patch.register_v1_envs, imageio, argparse, numpy as np

def play(model_path: str, env_id: str, outfile: str, episodes=1):
    from stable_baselines3.common.base_class import BaseAlgorithm
    model: BaseAlgorithm = BaseAlgorithm.load(model_path)

    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    for _ in range(episodes):
        obs, _info = env.reset()
        done = truncated = False
        while not (done or truncated):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, done, truncated, _info = env.step(action)
    env.close()
    imageio.mimsave(outfile, frames, fps=15)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env",   required=True)
    ap.add_argument("--out",   default="demo.mp4")
    play(**vars(ap.parse_args()))