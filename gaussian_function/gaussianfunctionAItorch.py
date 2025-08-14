import torch
import matplotlib.pyplot as plt

# prompt 2: Convert this script to PyTorch and to use its Tensors instead of Numpy

# Define the 2D Gaussian function using PyTorch
def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    normalization = 1 / (2 * torch.pi * sigma_x * sigma_y)
    exponent = -(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2))
    return normalization * torch.exp(exponent)

# Create a grid of (x, y) values using torch
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')  # Match NumPy's 'xy' behavior

# Compute the Gaussian values
Z = gaussian_2d(X, Y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1)

# Convert to NumPy for plotting with Matplotlib
Z_np = Z.numpy()
X_np = X.numpy()
Y_np = Y.numpy()

# Plot the 2D Gaussian using a contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_np, Y_np, Z_np, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Gaussian Function (PyTorch)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()