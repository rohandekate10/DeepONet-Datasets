import numpy as np
import sys
import time
from functools import wraps

def get_data(
    fname_train, fname_test, residual=False, stackbranch=False, stacktrunk=False
):
    N = 500 # High Fidelity Sample Trajectories
    # i = 0
    # idx = np.arange(i * N, (i + 1) * N)
    idx = np.random.choice(100000, size=N, replace=False)

    d = np.load(fname_train)
    X_branch = d["X0"][idx].astype(np.float32) # f
    X_trunk = d["X1"][idx].astype(np.float32) # x
    if stackbranch:
        X_branch = np.hstack((d["X0"][idx], d["y_low"][idx]))
    if stacktrunk:
        X_trunk = np.hstack((d["X1"][idx], d["y_low_x"][idx]))
    X_train = (X_branch, X_trunk) #(u_train, y_train)
    y_train = d["y"][idx].astype(np.float32) # u_HF
    if residual:
        y_train -= d["y_low_x"][idx] # x_LF

    d = np.load(fname_test)
    X_branch = d["X0"].astype(np.float32)
    X_trunk = d["X1"].astype(np.float32)
    if stackbranch:
        X_branch = np.hstack((d["X0"], d["y_low"]))
    if stacktrunk:
        X_trunk = np.hstack((d["X1"], d["y_low_x"]))
    X_test = (X_branch, X_trunk)
    y_test = d["y"].astype(np.float32)
    if residual:
        y_test -= d["y_low_x"]
    return X_train, y_train, X_test, y_test
#(u_train, y_train), G_train, (u_test, y_test), G_test = X_train, y_train, X_test, y_test

def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper