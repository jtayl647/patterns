import torch
import matplotlib.pyplot as plt

# prompt 1: Can you implement the Mandelbrot Set using PyTorch.

# Set device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
width, height = 1000, 1000  # resolution
max_iter = 100

# Ranges in the complex plane
x = torch.linspace(-2.0, 1.0, width, device=device)
y = torch.linspace(-1.5, 1.5, height, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')

# Complex grid: c = x + iy
C = X + 1j * Y
Z = torch.zeros_like(C)
div_time = torch.full(C.shape, max_iter, dtype=torch.int32, device=device)

# Mandelbrot iteration
for i in range(max_iter):
    mask = torch.abs(Z) <= 2.0
    Z[mask] = Z[mask] * Z[mask] + C[mask]
    div_time[mask & (torch.abs(Z) > 2.0)] = i  # first escape time

# Move to CPU for plotting
div_time_cpu = div_time.cpu()

# Plotting
plt.figure(figsize=(8, 8))
plt.imshow(div_time_cpu.T, cmap="magma", extent=[-2, 1, -1.5, 1.5])
plt.title("Mandelbrot Set (PyTorch)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.axis("off")
plt.show()