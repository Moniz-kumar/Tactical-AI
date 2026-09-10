from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from .env import TacticalFootballEnv

def train_agent(mode="inference",timesteps=50_000,seed=1,out_dir="models"):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    env=Monitor(TacticalFootballEnv(observation_mode=mode,seed=seed))
    model=MaskablePPO("MlpPolicy",env,learning_rate=3e-4,n_steps=1024,batch_size=64,gamma=.99,gae_lambda=.95,ent_coef=.01,seed=seed,verbose=1,tensorboard_log="runs/",policy_kwargs=dict(net_arch=dict(pi=[128,64],vf=[128,64])))
    model.learn(total_timesteps=timesteps)
    path=Path(out_dir)/f"{mode}_seed{seed}"; model.save(path); return str(path)+".zip"
