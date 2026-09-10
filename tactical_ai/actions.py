from dataclasses import dataclass
from itertools import product
from .config import FORMATIONS, LINE_HEIGHTS, PRESSING, TEMPOS

@dataclass(frozen=True)
class TacticalAction:
    formation: int
    line_height: int
    pressing: int
    tempo: int
    substitute: int

    def as_dict(self):
        return {
            "formation": FORMATIONS[self.formation],
            "line_height": LINE_HEIGHTS[self.line_height],
            "pressing": PRESSING[self.pressing],
            "tempo": TEMPOS[self.tempo],
            "substitute": bool(self.substitute),
        }

ACTION_TABLE = [
    TacticalAction(f, l, p, t, s)
    for f, l, p, t, s in product(
        range(len(FORMATIONS)),
        range(len(LINE_HEIGHTS)),
        range(len(PRESSING)),
        range(len(TEMPOS)),
        range(2),
    )
]
