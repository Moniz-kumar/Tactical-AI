from dataclasses import dataclass

STYLES = ("high_press", "possession", "counterattack")
FORMATIONS = ("4-3-3", "4-2-3-1", "3-5-2")
LINE_HEIGHTS = ("low", "mid", "high")
PRESSING = ("low", "medium", "high")
TEMPOS = ("slow", "balanced", "fast")

@dataclass(frozen=True)
class EnvConfig:
    decision_minutes: int = 5
    match_minutes: int = 90
    max_substitutions: int = 5
    shaping_weight: float = 0.08
    fatigue_penalty_weight: float = 0.01
    tactic_change_penalty: float = 0.002
    opponent_switch_probability: float = 0.08

    @property
    def max_steps(self) -> int:
        return self.match_minutes // self.decision_minutes
