from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .backend import MatchState
from .config import EnvConfig

@dataclass
class GRFBackendConfig:
    scenario:str="11_vs_11_stochastic"
    frame_skip:int=50
    render:bool=False
    logdir:str="grf_logs"
    control_all_left_players:bool=False

class TacticalLowLevelController:
    """Maps manager tactics to player-level GRF actions.

    GRF has no native change_formation command. Formation changes are realised
    by changing positional targets while the same match continues.
    """
    TEMPLATES={
        0:[(-.9,0),(-.62,-.34),(-.68,-.12),(-.68,.12),(-.62,.34),(-.28,-.22),(-.34,0),(-.28,.22),(.18,-.32),(.4,0),(.18,.32)],
        1:[(-.9,0),(-.62,-.34),(-.68,-.12),(-.68,.12),(-.62,.34),(-.34,-.14),(-.34,.14),(.02,-.3),(.06,0),(.02,.3),(.4,0)],
        2:[(-.9,0),(-.66,-.22),(-.72,0),(-.66,.22),(-.25,-.38),(-.3,-.16),(-.34,0),(-.3,.16),(-.25,.38),(.3,-.12),(.3,.12)]}
    def targets(self,action):
        shift={0:-.08,1:0,2:.1}[action.line_height]+{0:-.03,1:.02,2:.08}[action.pressing]
        return [(x if i==0 else np.clip(x+shift,-.92,.88),y) for i,(x,y) in enumerate(self.TEMPLATES[action.formation])]

class GRFBackend:
    def __init__(self,config:EnvConfig,grf:GRFBackendConfig|None=None):
        self.config=config; self.grf=grf or GRFBackendConfig(); self.controller=TacticalLowLevelController(); self._state=MatchState(); self._style="possession"
        try: import gfootball.env as football_env
        except ImportError as e: raise RuntimeError("Google Research Football is not installed. Use backend='toy' or install the dedicated GRF environment.") from e
        controls=11 if self.grf.control_all_left_players else 1
        self.env=football_env.create_environment(env_name=self.grf.scenario,representation="raw",rewards="scoring",render=self.grf.render,number_of_left_players_agent_controls=controls,number_of_right_players_agent_controls=0,action_set="v2")
        self._last_raw=None
    @property
    def state(self): return self._state
    @property
    def opponent_style(self): return self._style
    def reset(self,seed=None):
        self._last_raw=self.env.reset(); self._state=MatchState(); return self._state
    def advance(self,action):
        # Stage-2/3 integration boundary: manager action persists through a
        # decision interval. Full role-aware low-level action translation is
        # validated separately before final experiments.
        low=0
        for _ in range(self.grf.frame_skip):
            result=self.env.step(low)
            self._last_raw=result[0]
            if len(result)>=3 and result[2]: break
        self._state.minute=min(90,self._state.minute+self.config.decision_minutes)
        return self._state
