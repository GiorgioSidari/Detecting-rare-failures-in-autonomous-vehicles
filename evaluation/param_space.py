import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

PARAM_BOUNDS = {
    'names':  ['initial_speed', 'friction_coefficient', 'detection_distance', 'nominal_delay'],
    'lower':  np.array([5.0,  0.1, 10.0, 0.0]),
    'upper':  np.array([40.0, 0.9, 120.0, 0.8]),
}

def lhs_sample(n: int, bounds: dict, seed: int = 42) -> np.ndarray:
    sampler = LatinHypercube(d=4, seed=seed)
    unit_sample = sampler.random(n=n)
    return scale(unit_sample, bounds['lower'], bounds['upper'])


