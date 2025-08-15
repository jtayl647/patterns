import torch
import matplotlib.pyplot as plt

# Select GPU ("cuda") if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Generate Dragon Curve turns --------------------
def generate_turns(n):
    # Start with an empty tensor for turn sequence (int8 to save memory, -1 or +1 only)
    # Stored directly on the GPU (device)
    turns = torch.empty(0, dtype=torch.int8, device=device)

    # Iteratively build the Dragon Curve turn sequence
    for _ in range(n):
        # Reverse the current sequence and invert signs
        # This flips left↔right
        inverted = -turns.flip(0)  # flip() is GPU-accelerated for element-wise operations

        # Append: previous sequence + a left turn (1) + inverted reversed sequence
        # Left turn (1) is always inserted in the middle each iteration
        # "torch.cat" concatenates multiple tensors together (allocates new space)
        # Note: this is parallelized at the memory copy level on the GPU, 
        # but the loop itself is sequential
        turns = torch.cat([
            turns,
            torch.tensor([1], dtype=torch.int8, device=device),
            inverted
        ])
    # After n iterations, length = 2^n - 1
    return turns

# -------------------- Convert turns to 2D points in parallel --------------------
def generate_points(turns):
    # Add an initial "0" turn so the first step starts facing right (direction index 0)
    # Concatenates [0] and turns. Small setup, not the main parallelism
    dir_changes = torch.cat([
        torch.tensor([0], dtype=torch.int8, device=device),
        turns
    ])

    # Cumulative sum of direction changes gives the direction index at each step
    # modulo 4 keeps values in [0,1,2,3] meaning right, up, left, down
    # This is where real GPU parallelism starts: torch.cumsum is fully parallelized across the tensor
    directions = torch.remainder(torch.cumsum(dir_changes, dim=0), 4)

    # Lookup table mapping direction indices to (dx, dy) movement vectors
    # 0: right (1,0), 1: up (0,1), 2: left (-1,0), 3: down (0,-1)
    # This table itself is small and static — no parallelism here
    step_map = torch.tensor([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ], dtype=torch.int32, device=device)

    # Map each direction index to its corresponding step vector
    # Here the parallelism happens: indexing a large `directions` tensor
    # retrieves all (dx, dy) vectors in parallel on the GPU
    steps = step_map[directions]

    # Cumulative sum of step vectors gives all (x,y) coordinates along the curve
    # Fully parallel on GPU
    points = torch.cumsum(steps, dim=0)
    return points

# -------------------- Main --------------------
def main():
    n = 24  # Number of iterations → controls detail level and number of points

    # Generate turn sequence on GPU (sequential concatenation, some parallel memory ops)
    turns = generate_turns(n)

    # Convert turn sequence into all curve coordinates on GPU (fully vectorized, parallel)
    points = generate_points(turns)

    # Move final points to CPU for plotting (Matplotlib can't plot GPU tensors)
    points_cpu = points.cpu().numpy()

    # Plot the curve
    plt.figure(figsize=(8, 8))
    plt.plot(points_cpu[:, 0], points_cpu[:, 1], linewidth=0.5, color='purple')
    plt.axis('equal')  # Keep aspect ratio square
    plt.title("Dragon Curve (GPU Vectorized)")
    plt.show()

# Only run if script is executed directly
if __name__ == "__main__":
    main()