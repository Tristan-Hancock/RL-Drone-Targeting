import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from Drone3DEnv import Drone3DEnv

def make_env(render_mode=None):
    def _init():
        return Drone3DEnv(render_mode=render_mode)
    return _init

def train_agent(total_timesteps=200000):
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs/drone3d", exist_ok=True)
    
    vec_env = make_vec_env(make_env(render_mode=None), n_envs=4)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    eval_env = make_vec_env(make_env(render_mode=None), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./tb_logs/eval_logs/",
        eval_freq=10000,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_drone3d"
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tb_logs/drone3d/"
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save("./models/ppo_drone3d")
    vec_env.close()
    eval_env.close()
    print("Training complete and model saved.")
    return model

if __name__ == "__main__":
    train_agent()
