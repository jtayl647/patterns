import torch
import numpy as np
import matplotlib.pyplot as plt

# Set up the computing device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Convert list to tensor and send it to the selected device
    return torch.tensor(turns, device=device)

def generate_points(turns):
    """
    Convert the turn sequence into 2D points (x, y) that form the Dragon Curve path.
    
    We use a direction index (0–3) to track which direction we're facing:
        0 = right, 1 = up, 2 = left, 3 = down
    
    For each turn:
      - Update the direction index,
      - Move one step in that new direction,
      - Record the new position.
    """
    directions = torch.tensor([
        [1, 0],   # Right
        [0, 1],   # Up
        [-1, 0],  # Left
        [0, -1],  # Down
    ], device=device)

    dir_idx = 0  # Start facing right (index 0)
    pos = torch.tensor([0, 0], dtype=torch.int32, device=device)
    points = [pos.clone()]  # Start point at origin

    # Move one step forward before any turns
    pos = pos + directions[dir_idx]
    points.append(pos.clone())

    # Apply each turn to update direction and move
    for t in turns:
        # Update the direction index: left (+1), right (-1), wrap around using modulo 4
        dir_idx = (dir_idx + t.item()) % 4
        # Move in the new direction
        pos = pos + directions[dir_idx]
        # Save the new position
        points.append(pos.clone())

    # Stack list of points into a tensor of shape (N, 2)
    return torch.stack(points)

def main():
    """
    Main function:
      - Ask user for number of iterations (n),
      - Generate turn sequence and points,
      - Plot the curve using Matplotlib.
    """
    n = int(input("Enter number of iterations: "))
    turns = generate_turns(n)                      # Generate list of turn instructions
    points = generate_points(turns).cpu().numpy()  # Get points and move to CPU for plotting

    # Plot the curve
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], linewidth=0.5, color='purple')
    plt.axis('equal')  # Ensure x and y axes have the same scale
    plt.title(f"Dragon Curve (n={n})")
    plt.show()

# Run the program if this script is called directly
if __name__ == "__main__":
    main()