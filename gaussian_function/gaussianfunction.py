import torch
import numpy as np

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# DEMO QUESTIONS
# Change the Gaussian function into a 2D sine or cosine function (1 Mark)
# What do you get when you multiply both the Gaussian and the sine/cosine function together and
# visualise it? (1 Mark)

# Compute Gaussian: highest at the center and decay exponentially as you move away.
# z = torch.exp(-(x**2+y**2)/2.0) 

# Compute Sine: 2D wave pattern oscillates between -1 and 1.
# z = (torch.sin(x + y))

# Compute Gabor Filter: diagonal sine waves, strongest at the center, fade out toward the edges.
z = (torch.sin(x + y)) * (torch.exp(-(x**2+y**2)/2.0))

#plot
import matplotlib.pyplot as plt
plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.show()