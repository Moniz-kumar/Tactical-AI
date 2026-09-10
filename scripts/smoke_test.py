from tactical_ai.env import TacticalFootballEnv
from tactical_ai.baselines import rule_based_policy

env=TacticalFootballEnv(observation_mode="inference")
obs,info=env.reset(seed=42); total=0.0; done=False
while not done:
    action=rule_based_policy(env)
    obs,reward,terminated,truncated,info=env.step(action)
    total+=reward; print(env.render()); done=terminated or truncated
print("\nFinal score:",info["score"])
print("Final xG:",info["xg"])
print("Reward:",round(total,4))
print("Final opponent belief:",info["opponent_belief"])
