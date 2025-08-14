import torch
import numpy as np

# Device configuration: use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Given point
# x_min, x_max = -2, 1
# y_min, y_max = -1.3, 1.3
# step = 0.0005

# My zoomed point.
x_min, x_max = -1.405, -1.395
y_min, y_max = -0.005, 0.005
step = 0.000002 

# Create the grid
Y, X = np.mgrid[y_min:y_max:step, x_min:x_max:step]

# Convert the NumPy arrays to PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# Combine x and y into a 2D array of complex numbers: z = x + iy
z = torch.complex(x, y)  # important for Julia/Mandelbrot iteration

# Make a copy of z for iterative calculations
zs = z.clone()  # zs will be updated at each iteration

# Initialize a tensor to count the number of iterations before divergence
ns = torch.zeros_like(z)

# Julia constant (used if you want to generate Julia set instead)
c = complex(-0.8, 0.156)

# Move all tensors to the GPU (if available) for faster computation
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# -------------------- Fractal computation --------------------
# Iteratively compute the Mandelbrot (or Julia) function
for i in range(200):
    # Compute the new value of zs for this iteration
    # Mandelbrot: zs^2 + z
    zs_ = zs*zs + z  
    # Julia: uncomment the next line instead of the Mandelbrot line
    # zs_ = zs*zs + c  

    # Check which points have not diverged yet
    # A point is considered diverged if its magnitude is >= 4
    not_diverged = torch.abs(zs_) < 4.0

    # Increment the iteration count for points that have not diverged
    ns += not_diverged

    # Update zs for the next iteration
    zs = zs_

# -------------------- Visualization --------------------
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """
    Convert the iteration counts into a colorful RGB image.
    
    a: 2D array of iteration counts
    """
    # Map iteration counts to angles for sine/cosine coloring
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])  # add a color channel dimension

    # Create RGB channels with different trigonometric mappings
    img = np.concatenate([
        10 + 20*np.cos(a_cyclic),   # Red channel
        30 + 50*np.sin(a_cyclic),   # Green channel
        155 - 80*np.cos(a_cyclic)   # Blue channel
    ], 2)

    # Set points that never diverged to black
    img[a==a.max()] = 0

    # Clip values to [0,255] and convert to uint8 for image display
    a = np.uint8(np.clip(img, 0, 255))
    return a    

# Display the fractal image
plt.imshow(processFractal(ns.cpu().numpy()))  # move tensor back to CPU and convert to NumPy
plt.tight_layout(pad=0)
plt.show()