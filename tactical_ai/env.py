from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig, STYLES
from .actions import ACTION_TABLE
from .backend import ToyMatchBackend
from .bayes import OpponentStyleFilter

class TacticalFootballEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, observation_mode="inference", config=None, seed=None, backend="toy", backend_kwargs=None):
        super().__init__()
        if observation_mode not in {"blind","inference","full_info"}: raise ValueError("invalid observation_mode")
        self.observation_mode=observation_mode; self.config=config or EnvConfig(); backend_kwargs=backend_kwargs or {}
        if backend == "toy": self.backend=ToyMatchBackend(self.config)
        elif backend == "grf":
            from .grf_backend import GRFBackend, GRFBackendConfig
            grf_cfg=backend_kwargs.pop("grf_config",None)
            if isinstance(grf_cfg,dict): grf_cfg=GRFBackendConfig(**grf_cfg)
            self.backend=GRFBackend(self.config,grf=grf_cfg)
        else: raise ValueError("backend must be 'toy' or 'grf'")
        self.backend_name=backend; self.filter=OpponentStyleFilter(); self._initial_seed=seed
        self.action_space=spaces.Discrete(len(ACTION_TABLE))
        style_dim=0 if observation_mode=="blind" else len(STYLES)
        self.observation_space=spaces.Box(low=-1.0,high=1.0,shape=(11+style_dim,),dtype=np.float32)
        self.substitutions_used=0; self.current_action=ACTION_TABLE[0]; self.previous_action_index=0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed); seed=self._initial_seed if seed is None else seed
        self.backend.reset(seed); self.filter.reset(); self.substitutions_used=0
        self.current_action=ACTION_TABLE[0]; self.previous_action_index=0
        return self._get_obs(),self._get_info()

    def action_masks(self):
        valid=np.ones(self.action_space.n,dtype=bool)
        if self.substitutions_used>=self.config.max_substitutions:
            for i,a in enumerate(ACTION_TABLE):
                if a.substitute: valid[i]=False
        return valid

    def _get_evidence(self):
        s=self.backend.state
        return np.array([s.possession,s.pressure,s.territory,s.opponent_directness],dtype=float)

    def _style_features(self):
        if self.observation_mode=="blind": return np.array([],dtype=np.float32)
        if self.observation_mode=="inference": return self.filter.belief.astype(np.float32)
        onehot=np.zeros(len(STYLES),dtype=np.float32); onehot[STYLES.index(self.backend.opponent_style)]=1.0; return onehot

    def _get_obs(self):
        s=self.backend.state; a=self.current_action
        score_diff=np.clip((s.goals_for-s.goals_against)/5.0,-1,1)
        time_remaining=np.clip((self.config.match_minutes-s.minute)/self.config.match_minutes,0,1)
        subs_left=(self.config.max_substitutions-self.substitutions_used)/self.config.max_substitutions
        base=np.array([score_diff,time_remaining,2*s.fatigue-1,2*s.possession-1,2*s.territory-1,2*s.pressure-1,a.formation-1.0,a.line_height-1.0,a.pressing-1.0,a.tempo-1.0,2*subs_left-1],dtype=np.float32)
        return np.concatenate([base,self._style_features()]).astype(np.float32)

    def _get_info(self):
        s=self.backend.state
        return {"minute":s.minute,"score":(s.goals_for,s.goals_against),"xg":(round(s.xg_for,3),round(s.xg_against,3)),"opponent_style_true":self.backend.opponent_style,"opponent_style_estimate":self.filter.predicted_style,"opponent_belief":dict(zip(STYLES,self.filter.belief.round(4))),"substitutions_used":self.substitutions_used,"tactics":self.current_action.as_dict()}

    def step(self,action_index):
        action_index=int(action_index)
        if not self.action_masks()[action_index]: raise ValueError("Illegal substitution action")
        action=ACTION_TABLE[action_index]
        if action.substitute:
            self.substitutions_used+=1; self.backend.state.fatigue=max(0.0,self.backend.state.fatigue-.055)
        old_xg=self.backend.state.xg_for-self.backend.state.xg_against
        self.backend.advance(action); self.current_action=action; self.filter.update(self._get_evidence())
        s=self.backend.state; reward=self.config.shaping_weight*((s.xg_for-s.xg_against)-old_xg)
        reward-=self.config.fatigue_penalty_weight*max(0.0,s.fatigue-.75)
        if action_index!=self.previous_action_index: reward-=self.config.tactic_change_penalty
        terminated=s.minute>=self.config.match_minutes
        if terminated:
            diff=s.goals_for-s.goals_against; reward += 1.0 if diff>0 else (-1.0 if diff<0 else 0.0)
        self.previous_action_index=action_index
        return self._get_obs(),float(reward),terminated,False,self._get_info()

    def render(self):
        i=self._get_info(); return f"{i['minute']:02d}' | {i['score'][0]}-{i['score'][1]} | opp={i['opponent_style_estimate']} | belief={i['opponent_belief']} | tactics={i['tactics']}"
