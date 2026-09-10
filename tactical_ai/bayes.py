from __future__ import annotations
import numpy as np
from .config import STYLES

class OpponentStyleFilter:
    """Discrete Bayesian filter over hidden opponent tactical styles.

    Evidence vector:
        possession_share, pressure, territory, directness

    Likelihoods are simple Gaussian prototypes for the MVP. They should later
    be fitted from labelled simulated trajectories.
    """

    PROTOTYPES = {
        "high_press": np.array([0.50, 0.82, 0.62, 0.55]),
        "possession": np.array([0.64, 0.48, 0.66, 0.38]),
        "counterattack": np.array([0.40, 0.38, 0.42, 0.82]),
    }

    SIGMA = np.array([0.12, 0.13, 0.13, 0.14])

    def __init__(self, transition_stay: float = 0.90):
        self.transition_stay = transition_stay
        n = len(STYLES)
        off = (1.0 - transition_stay) / (n - 1)
        self.transition = np.full((n, n), off, dtype=float)
        np.fill_diagonal(self.transition, transition_stay)
        self.reset()

    def reset(self):
        self.belief = np.full(len(STYLES), 1.0 / len(STYLES), dtype=float)
        return self.belief.copy()

    def _likelihood(self, evidence: np.ndarray) -> np.ndarray:
        evidence = np.asarray(evidence, dtype=float)
        vals = []
        for style in STYLES:
            mu = self.PROTOTYPES[style]
            z = (evidence - mu) / self.SIGMA
            vals.append(np.exp(-0.5 * np.sum(z * z)))
        vals = np.asarray(vals, dtype=float)
        return vals + 1e-12

    def update(self, evidence):
        prior = self.belief @ self.transition
        likelihood = self._likelihood(np.asarray(evidence))
        posterior = prior * likelihood
        total = posterior.sum()
        self.belief = posterior / total if total > 0 else np.full(len(STYLES), 1/len(STYLES))
        return self.belief.copy()

    @property
    def predicted_style(self):
        return STYLES[int(np.argmax(self.belief))]
