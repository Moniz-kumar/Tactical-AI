import argparse
from tactical_ai.training import train_agent
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["blind","inference","full_info"],default="inference"); p.add_argument("--timesteps",type=int,default=50_000); p.add_argument("--seed",type=int,default=1); args=p.parse_args()
print("Saved:",train_agent(args.mode,args.timesteps,args.seed))
