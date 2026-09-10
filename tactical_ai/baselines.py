from __future__ import annotations
import numpy as np
from .actions import ACTION_TABLE

def static_balanced_policy(env):
    for idx,a in enumerate(ACTION_TABLE):
        if (a.formation,a.line_height,a.pressing,a.tempo,a.substitute)==(1,1,1,1,0): return idx
    return 0

def random_legal_policy(env):
    return int(env.np_random.choice(np.flatnonzero(env.action_masks())))

def rule_based_policy(env):
    s=env.backend.state; style_idx=int(np.argmax(env.filter.belief))
    target={0:(1,1,1,2),1:(1,1,2,1),2:(1,0,1,1)}[style_idx]
    do_sub=int(s.minute>=60 and s.fatigue>.70 and env.substitutions_used<env.config.max_substitutions)
    for idx,a in enumerate(ACTION_TABLE):
        if (a.formation,a.line_height,a.pressing,a.tempo,a.substitute)==(*target,do_sub): return idx
    return static_balanced_policy(env)
