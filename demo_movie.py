import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from time import time
from spring import SpringCoupledMasses

# define system and solve
system = SpringCoupledMasses(
    m=[1, 1, 1],
    pairs=[(0, 1), (0, 2), (1, 2)],
    stiffness=[1, 1, 1],
    damping=[0.5, 0.5, 0.5],
)
system.set_initial_condition(
    x0=np.array([[0.75, -0.5], [0, 0.5], [-0.75, -0.5]]),
    v0=np.array([[0, 0.1], [0, 0], [0, 0]]),
)
start = time()
system.rk4(T=5, timesteps=50001, frames=101)
print(f"Solved in {time() - start:.2f} s")


# make video
folder_path = "frames"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
for frame in range(101):
    for p in range(len(system.m)):
        plt.plot(system.X[frame, p, 0], system.X[frame, p, 1], "o", label=f"m{p}")
    for pi, pj in system.pairs:
        plt.plot(
            system.X[frame, (pi, pj), 0],
            system.X[frame, (pi, pj), 1],
            "--",
            color="grey",
            label=f"m{p}",
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig(f"{folder_path}/{frame:03d}.png")
    plt.close()
print(f"Saved frames to {folder_path}")

# Sort the PNG images by filename
png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Create a list to hold the images
images = []
for png_file in png_files:
    # Open each PNG image and append it to the list
    image = Image.open(os.path.join(folder_path, png_file))
    images.append(image)

# Save the list of images as a GIF
images[0].save("demo.gif", save_all=True, append_images=images[1:], duration=0, loop=0)
print("Generated gif")
