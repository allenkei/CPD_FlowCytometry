import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

##############################################
# ========== 1. Generate X (low noise) =======
##############################################
def generate_x(T=200, N=100, dx=3, noise_scale=0.1, global_noise=None):
    """
    Generate structured covariates X with reduced random noise.
    Uses shared random basis to keep replicates consistent.

    Parameters
    ----------
    T : int
        Number of time points
    N : int
        Number of samples per time
    dx : int
        Dimension of X
    noise_scale : float
        Magnitude of added Gaussian noise
    global_noise : np.ndarray or None
        Optional (T,N) shared noise pattern across replicates
    """
    t = np.linspace(0, 1, T)
    n_groups = dx // 3
    X = np.zeros((T, N, dx))

    if global_noise is None:
        # Shared noise pattern → helps make different replicates look similar
        global_noise = np.random.randn(T, N)

    for g in range(n_groups):
        inc = t[:, None] + noise_scale * global_noise
        dec = (1 - t)[:, None] + noise_scale * global_noise
        wave = (np.sin(8 * np.pi * t)[:, None] * np.exp(-5 * (t - 0.5)**2)[:, None]
                + noise_scale * global_noise)
        X[:, :, 3*g + 0] = inc
        X[:, :, 3*g + 1] = dec
        X[:, :, 3*g + 2] = wave
    return X


##############################################
# ========== 2. Softmax utility ==============
##############################################
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


##############################################
# ========== 3. Generate Y (low-noise GMM) ===
##############################################
def generate_gmm_data_segments(
    X, mean_funcs_list, weight_funcs_list,
    change_points=[51, 101, 151], K=3, dy=3,
    mean_jitter=0.05, cov_scale=0.5
):
    """
    Generate time-series data with segment-specific GMM structure (reduced noise).

    - Adds small jitter to component means per segment (mean_jitter)
    - Reduces covariance magnitude (cov_scale < 1)

    Parameters
    ----------
    X : np.ndarray
        Shape (T, N, dx)
    mean_funcs_list : list[list[callable]]
        List of mean functions for each segment
    weight_funcs_list : list[callable]
        List of mixture weight functions for each segment
    change_points : list[int]
        Change points marking start of next segment
    K : int
        Number of mixture components
    dy : int
        Dimension of Y
    mean_jitter : float
        Magnitude of random mean perturbation per segment
    cov_scale : float
        Covariance scaling (<1 makes clusters tighter)
    """
    T, N, dx = X.shape
    S = len(change_points) + 1
    cps = [0] + change_points + [T]

    Y = np.zeros((T, N, dy))
    comp_labels = np.zeros((T, N), dtype=int)

    for s in range(S):
        mean_funcs = mean_funcs_list[s]
        weight_func = weight_funcs_list[s]

        # small random jitter per segment (helps realism, keeps replicates consistent)
        seg_jitter = mean_jitter * np.random.randn(K, dy)

        for t in range(cps[s], cps[s+1]):
            for n in range(N):
                x = X[t, n]
                mus = [np.asarray(f(x)) + seg_jitter[k] for k, f in enumerate(mean_funcs)]
                pis = np.array(weight_func(x))
                pis /= pis.sum()
                k = np.random.choice(K, p=pis)
                comp_labels[t, n] = k
                cov = cov_scale * np.eye(dy)
                Y[t, n] = np.random.multivariate_normal(mus[k], cov)
    return X, Y, comp_labels


##############################################
# ========== 4. Visualize Y in 3D ============
##############################################
def visualize_Y_grid(Y, step=20, ncols=4, alpha=0.6, labels=None):
    """
    Visualize Y (T,N,3) snapshots as a grid of 3D scatter subplots.
    Each class label has a different color if labels is provided.
    """
    T, N, dy = Y.shape
    assert dy == 3, f"Need dy=3 for 3D plot, got dy={dy}"

    time_idx = np.arange(0, T, step)
    nplots = len(time_idx)
    nrows = int(np.ceil(nplots / ncols))

    y_min = Y.reshape(-1, 3).min(axis=0)
    y_max = Y.reshape(-1, 3).max(axis=0)
    ranges = list(zip(y_min, y_max))

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    cmap = cm.get_cmap("tab10")

    for i, t in enumerate(time_idx):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        if labels is not None:
            lbl_t = labels[t].astype(int)
            colors = [cmap(l % 10) for l in lbl_t]
            ax.scatter(
                Y[t, :, 0], Y[t, :, 1], Y[t, :, 2],
                c=colors, s=10, alpha=alpha
            )
        else:
            ax.scatter(
                Y[t, :, 0], Y[t, :, 1], Y[t, :, 2],
                color="royalblue", s=10, alpha=alpha
            )

        ax.set_title(f"t = {t+1}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=20, azim=30)
        ax.set_xlim(ranges[0]); ax.set_ylim(ranges[1]); ax.set_zlim(ranges[2])
        ax.set_xlabel("$Y_1$"); ax.set_ylabel("$Y_2$"); ax.set_zlabel("$Y_3$")

    plt.suptitle("3D Snapshots of Y (colored by comp_label)", fontsize=16)
    plt.tight_layout()
    plt.show()


##############################################
# ========== 5. Visualize X timeseries =======
##############################################
def visualize_X_timeseries(X, n_samples_show=100, figsize=(10, 6)):
    """
    Visualize each dimension of X over time, with mean ± std shading.
    """
    T, N, dx = X.shape
    t = np.arange(T)

    fig, axes = plt.subplots(dx, 1, figsize=figsize, sharex=True)
    if dx == 1:
        axes = [axes]

    for d in range(dx):
        ax = axes[d]
        mean_d = X[:, :, d].mean(axis=1)
        std_d = X[:, :, d].std(axis=1)

        idx_show = np.random.choice(N, min(n_samples_show, N), replace=False)
        for n in idx_show:
            ax.plot(t, X[:, n, d], color="lightgray", alpha=0.3, lw=0.8)

        ax.plot(t, mean_d, color="royalblue", lw=2, label=f"$X_{d+1}$ mean")
        ax.fill_between(t, mean_d - std_d, mean_d + std_d,
                        color="royalblue", alpha=0.2, label="±1 SD")

        ax.set_ylabel(f"$X_{d+1}$")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.3, linestyle="--")

    axes[-1].set_xlabel("Time (t)")
    plt.suptitle("Temporal Evolution of X (mean ± std)", fontsize=14)
    plt.tight_layout()
    plt.show()
