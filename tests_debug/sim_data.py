import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting

# Setup
torch.manual_seed(42)
T = 100
n1 = 100
n2 = 100
dim = 3  # NOW in 3D

# Means and covariances
mean1_before = torch.tensor([0.0, -2.0, 1.0])
mean2_before = torch.tensor([1.0, 2.0, -1.0])
mean1_after = torch.tensor([2.0,  2.0, 2.0])
mean2_after = torch.tensor([-1.0, -2.0, -3.0])
cov = torch.eye(dim) * 0.5

# Generate data
time_series_data = []
for t in range(1, T + 1):
    if t <= 50:
        data1 = torch.distributions.MultivariateNormal(mean1_before, cov).sample((n1,))
        data2 = torch.distributions.MultivariateNormal(mean2_before, cov).sample((n2,))
    else:
        data1 = torch.distributions.MultivariateNormal(mean1_after, cov).sample((n1,))
        data2 = torch.distributions.MultivariateNormal(mean2_after, cov).sample((n2,))
    
    combined_data = torch.cat([data1, data2], dim=0)  # Shape: (200, 3)
    time_series_data.append(combined_data)

# Save data
torch.save(time_series_data, "data/gaussian_time_series.pt")

# Plot a few time points in 3D
fig = plt.figure(figsize=(18, 6))
times_to_plot = list(range(10, 101, 10))

for idx, t in enumerate(times_to_plot):
    ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
    data = time_series_data[t - 1]
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], alpha=0.5, label="Cluster 1")
    ax.scatter(data[n1:, 0], data[n1:, 1], data[n1:, 2], alpha=0.5, label="Cluster 2")
    ax.set_title(f"Time t = {t}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.grid(True)

plt.tight_layout()
plt.savefig("data/gaussian_time_series.pdf")
plt.close()
