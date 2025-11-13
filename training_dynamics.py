import numpy as np
from models.dynamics_ensemble import Ensemble
from collect_dataset import collect_seed_data

def pack_xy(replay, obs_dim, mu_feat_dim, act_dim):
    X, Y = [], []

    for row in replay:
        x = row["x"]
        y = row["y"]

        X.append(x)
        Y.append(y)

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    assert X.shape[1] == obs_dim + mu_feat_dim + act_dim

    return X, Y

replay = collect_seed_data()
