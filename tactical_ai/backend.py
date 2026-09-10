from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import STYLES, EnvConfig
from .actions import TacticalAction

@dataclass
class MatchState:
    minute: int = 0
    goals_for: int = 0
    goals_against: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    possession: float = 0.50
    territory: float = 0.50
    pressure: float = 0.50
    fatigue: float = 0.05
    opponent_directness: float = 0.50

class MatchBackend:
    def reset(self, seed=None): ...
    def advance(self, action: TacticalAction): ...
    @property
    def state(self) -> MatchState: ...
    @property
    def opponent_style(self) -> str: ...

class ToyMatchBackend(MatchBackend):
    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng()
        self._state = MatchState()
        self._style = "possession"

    @property
    def state(self): return self._state
    @property
    def opponent_style(self): return self._style

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._state = MatchState()
        self._style = self.rng.choice(STYLES)
        return self._state

    def _maybe_switch_opponent(self):
        if self.rng.random() < self.config.opponent_switch_probability:
            self._style = self.rng.choice([s for s in STYLES if s != self._style])

    def advance(self, action: TacticalAction):
        self._maybe_switch_opponent()
        press, line, tempo = action.pressing / 2.0, action.line_height / 2.0, action.tempo / 2.0
        opp = {
            "high_press": dict(pressure=.82, possession=.50, territory=.62, directness=.55, attack=.62),
            "possession": dict(pressure=.48, possession=.64, territory=.66, directness=.38, attack=.58),
            "counterattack": dict(pressure=.38, possession=.40, territory=.42, directness=.82, attack=.66),
        }[self._style]
        matchup = 0.0
        if self._style == "high_press":
            matchup += .10 if tempo >= .5 else -.06
            matchup += .05 if line < .75 else -.04
        elif self._style == "possession": matchup += .09 if press >= .75 else -.04
        else:
            matchup += .10 if line <= .5 else -.10
            matchup += .04 if tempo <= .5 else -.02
        formation_effect = {0:.025,1:.01,2:-.005}[action.formation]
        fatigue = self._state.fatigue
        attack_strength = .48 + .12*tempo + .06*press + matchup + formation_effect - .16*fatigue
        defend_strength = .50 + .08*(1-line) + .04*press - max(0,line-.5)*(.12 if self._style=="counterattack" else .02)
        dt = self.config.decision_minutes / 90.0
        xgf = max(.005, dt*(1.15+attack_strength+self.rng.normal(0,.08)))
        xga = max(.005, dt*(1.15+opp["attack"]-defend_strength+self.rng.normal(0,.08)))
        gf, ga = int(self.rng.poisson(xgf)), int(self.rng.poisson(xga))
        possession=np.clip(.50+.10*(press-.5)+.10*(1-tempo)-.22*(opp["possession"]-.5)+self.rng.normal(0,.04),.20,.80)
        territory=np.clip(.48+.14*line+.08*press-.16*(opp["territory"]-.5)+self.rng.normal(0,.04),.15,.85)
        fatigue_gain=.018+.025*press+.015*tempo
        s=self._state
        s.fatigue=float(np.clip(s.fatigue+fatigue_gain,0,1)); s.minute += self.config.decision_minutes
        s.goals_for += gf; s.goals_against += ga; s.xg_for += xgf; s.xg_against += xga
        s.possession=float(possession); s.territory=float(territory)
        s.pressure=float(np.clip(opp["pressure"]+self.rng.normal(0,.04),0,1))
        s.opponent_directness=float(np.clip(opp["directness"]+self.rng.normal(0,.04),0,1))
        return s
