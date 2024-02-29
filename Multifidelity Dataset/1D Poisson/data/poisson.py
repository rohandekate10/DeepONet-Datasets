import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
from tqdm import trange

from .spaces import GRF
from .data_utils import timing


def solver(f, N):
    """u_xx = 20f, x \in [0, 1]
    u(0) = u(1) = 0
    """
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h ** 2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


def example(plot_figures=True):
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100

    features = space.random(1)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])
    y = solver(sensor_values[0], m)
    np.savetxt("./data/poisson_high.dat", np.vstack((sensors, np.ravel(sensor_values), y)).T)

    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = solver(f_low[0], m_low)
    np.savetxt("./data/poisson_low.dat", np.vstack((x_low, y_low)).T)

    if plot_figures:
        fig,(ax1,ax2) = plt.subplots(2,1, figsize=(8, 10))
        ax1.plot(sensors,sensor_values[0], "-." ,label="Actual $f$ for High-Fidelity")
        ax1.plot(sensors,y, "-o" ,label="High-Fidelity Solution")
        ax1.legend()
        ax1.grid()
        ax1.set_title(f"High-Fidelity Dataset Example @$\Delta x = 1/99$")
        ax1.set_xlabel(f"x")
        ax1.set_ylabel("$u_{HF}$")

        ax2.plot(x_low,f_low[0],"-.",label="Actual $f$ for Low-Fidelity")
        ax2.plot(x_low,y_low,"-o",label="Low-Fidelity Solution")
        ax2.legend()
        ax2.grid()
        ax2.set_title(f"Low-Fidelity Dataset Example @$\Delta x = 1/9$")
        ax2.set_xlabel(f"x")
        ax2.set_ylabel("$u_{LF}$")
        plt.tight_layout()
        plt.show()


@timing
def gen_data(fname="train", plot_figures=False):
    print("Generating operator data...", flush=True)
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100
    num = 100000

    features = space.random(num)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])

    x = []
    y = []
    for i in trange(num):
        tmp = solver(sensor_values[i], m)
        idx = np.random.randint(0, m, size=1)
        x.append(sensors[idx])
        y.append(tmp[idx])
    x = np.array(x)
    y = np.array(y)

    print("Shape of f_{HF}: " + f"{sensor_values.shape}")
    print("Shape of x_{HF}: " + f"{x.shape}")
    print("Shape of u_{HF}: " + f"{y.shape}")
    print(f"\n*************************************************\n")


    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = []
    y_low_x = []
    for i in trange(num):
        tmp = solver(f_low[i], m_low)
        tmp = interpolate.interp1d(x_low, tmp, copy=False, assume_sorted=True)
        y_low.append(tmp(sensors))
        y_low_x.append(tmp(x[i]))
    y_low = np.array(y_low)
    y_low_x = np.array(y_low_x)
    np.savez_compressed(
        f"./data/{fname}.npz", X0=sensor_values, X1=x, y=y, y_low=y_low, y_low_x=y_low_x
    )

    print("Shape of f_{LF}: " + f"{f_low.shape}")
    print("Shape of x_{LF}: " + f"{y_low_x.shape}")
    print("Shape of u_{LF}: " + f"{y_low.shape}")

    if plot_figures:
        fig, axes = plt.subplots(5, 1, figsize=(8, 20))
        # for i in range(5):
        for i, ax in enumerate(axes.flat):
            ax.plot(sensors, sensor_values[i], "k", label = "Randomly Sampled $f$")
            ax.plot(x[i], y[i], "or", label="High-Fidelity Point $u_{HF}$")
            ax.plot(sensors, y_low[i], "b", label = "Low-Fidelity Sample")
            ax.plot(x[i], y_low_x[i], "xb", label="Low-Fidelity Evaluation Point $x_{LF}$")
            ax.set_xlabel(f"$x$")
            ax.set_ylabel(f"$f$")
            ax.legend()
            ax.grid()
        
        plt.tight_layout()
        plt.show()