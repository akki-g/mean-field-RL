from pettingzoo.mpe import simple_spread_v3
import numpy as np

def wrap_torus(xy):
    return np.mod(xy, 1.0)

def make_env(N=5, local_ratio=0.5, seed=0):
    env = simple_spread_v3.parallel_env(
        N=N, local_ratio=local_ratio, max_cycles=50, continuous_actions=True
    )

    env.reset(seed=seed)
    return env

def position_to_mu(positions, bins=50):
    pos = wrap_torus(np.asarray(positions))
    H, edges = np.histogramdd(pos, bins=bins, range=[[0,1], [0,1]], density=True)

    H = np.clip(H, 1e-12, None)
    dx = dy = 1.0/bins

    entropy = -np.sum(H * np.log(H)) * dx * dy

    return {"grid_pdf": H, "entropy": entropy}


def h_C(mu_info, C_threshold):
    return mu_info["entropy"] - C_threshold