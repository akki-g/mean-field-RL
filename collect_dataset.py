import numpy as np
import torch
from collections import deque
from envs.mpe_spread import make_env
from typing import Dict, Any


def wrap01(xy):
    return np.mod((xy+1.0) / 2.0, 1.0)

def soft_hist2d_numpy(positions, bins=50):
    pos01 = wrap01(np.asarray(positions))
    H, edges = np.histogramdd(pos01, bins=bins, range=[[0,1],[0,1]], density=True)
    H = np.clip(H, 1e-12, None)
    dx = dy = 1.0 / bins
    Hmu = -np.sum(H * np.log(H)) * dx * dy

    return H, Hmu

def mu_features(H, Hmu, k=4):
    flat = H.reshape(-1)
    topk = np.sort(flat)[-k:]
    return np.concatenate(([Hmu], topk), axis=0).astype(np.float32)


def collect_seed_data(episodes=10, max_cycles=100, bins=50, topk=4, seed=0):
    env = make_env(seed=seed)
    replay = []

    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        obs = env.reset(seed=seed+ep)
        done = {a: False for a in env.agents}

        for t in range(max_cycles):
            
            positions = [obs[a][0:2] for a in env.agents]
            H, Hmu = soft_hist2d_numpy(positions, bins=bins)
            phi_mu = mu_features(H, Hmu, k=topk)

            acts = {
                a: rng.uniform(-1,1, size=env.action_space(a).shape)
                for a in env.agents
            }

            next_obs, rew, term, trunc, info = env.step(acts)

            for a in env.agents:
                s_t = obs[a].astype(np.float32)
                a_t = acts[a].astype(np.float32)
                y = next_obs[a].astype(np.float32)
                x = np.concatenate([s_t, phi_mu, a_t], 0)
                replay.append({
                    "x": x, "y": y,
                    "entropy": Hmu, "agent": a
                })
            
            obs = next_obs
            if all(term.values()) or all(trunc.values()):
                break
    
    return replay

