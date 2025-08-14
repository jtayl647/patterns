import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Device configuration --------------------
# Set up the computing device: GPU if available, otherwise CPU
# This allows major computations (tensors, arithmetic) to leverage GPU parallelism
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Generate Dragon Curve turn sequence --------------------
def generate_turns(n):
    """
    Generate a list of turns to draw the Dragon Curve.

    The Dragon Curve is built iteratively. At each step:
      - Take the current sequence of turns,
      - Add a left turn (1),
      - Add the reversed and inverted version of the previous turns.

    Each turn is either:
      - 1  = turn left (90°),
      - -1 = turn right (90°).

    Example for n = 2:
      Iteration 1: [1]
      Iteration 2: [1, 1, -1]
    """
    turns = []
    for _ in range(n):
        # Copy and reverse previous turn sequence
        reversed_turns = list(reversed(turns))
        # Invert the turn directions (left becomes right and vice versa)
        inverted = []
        for t in reversed_turns:
            inverted.append(-t)
        # Build new turn sequence: previous turns + left turn + inverted reverse
        turns = turns + [1] + inverted
    # Convert list to PyTorch tensor and move to selected device (GPU/CPU)
    # PyTorch tensors allow potential parallelism and efficient memory management
    return torch.tensor(turns, device=device)

# -------------------- Convert turn sequence to 2D points --------------------
def generate_points(turns):
    """
    Convert the turn sequence into 2D points (x, y) forming the Dragon Curve path.

    Uses a direction index (0–3) to track which direction we're facing:
        0 = right, 1 = up, 2 = left, 3 = down

    Each turn updates the direction and moves one step in that direction.
    """
    # Define unit movement vectors for directions: right, up, left, down
    directions = torch.tensor([
        [1, 0],   # Right
        [0, 1],   # Up
        [-1, 0],  # Left
        [0, -1],  # Down
    ], device=device)

    dir_idx = 0  # Start facing right
    pos = torch.tensor([0, 0], dtype=torch.int32, device=device)  # Start at origin
    points = [pos.clone()]  # Record start point

    # Move one step forward before applying any turns
    pos = pos + directions[dir_idx]
    points.append(pos.clone())

    # Apply each turn to update direction and move
    for t in turns:
        # Update the direction index: left (+1), right (-1), wrap using modulo 4
        dir_idx = (dir_idx + t.item()) % 4
        # Move in the new direction
        pos = pos + directions[dir_idx]
        # Save the new position
        points.append(pos.clone())

    # Stack all points into a PyTorch tensor of shape (N, 2)
    # Using PyTorch here enables efficient memory management for millions of points
    return torch.stack(points)

# -------------------- Main function to run the Dragon Curve --------------------
def main():
    """
    Main workflow:
      1. Generate turn sequence for n iterations
      2. Compute 2D points from the turns
      3. Plot the resulting Dragon Curve fractal
    """
    n = 21  # Number of iterations; higher n → more detailed fractal
    turns = generate_turns(n)                      # PyTorch tensor on GPU
    points = generate_points(turns).cpu().numpy()  # Convert to CPU/NumPy for plotting

    # -------------------- Plotting --------------------
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], linewidth=0.5, color='purple')
    plt.axis('equal')  # Equal aspect ratio to preserve fractal shape
    plt.title(f"Dragon Curve")
    plt.show()

# Run program if this script is executed directly
if __name__ == "__main__":
    main()