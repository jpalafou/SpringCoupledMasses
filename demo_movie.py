import matplotlib.pyplot as plt
import os
from PIL import Image
from time import time
from spring import SpringCoupledMasses

# define system and solve
system = SpringCoupledMasses(
    m=[1, 1, 1],
    pairs=[(0, 1), (0, 2), (1, 2)],
    stiffness=[2, 2, 2],
    damping=[0.1, 0.1, 0.1],
)
system.set_initial_condition(d=3)
start = time()
system.rk4(T=5, timesteps=50001, frames=101)
print(f"Solved in {time() - start:.2f} s")

# make 3d plots for frames of the gif
folder_path = "frames"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
for frame in range(101):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for p in range(len(system.m)):
        ax.plot(
            system.X[frame, p, 0],
            system.X[frame, p, 1],
            system.X[frame, p, 2],
            "o",
            label=f"m{p}",
        )
    for pi, pj in system.pairs:
        ax.plot(
            system.X[frame, (pi, pj), 0],
            system.X[frame, (pi, pj), 1],
            system.X[frame, (pi, pj), 2],
            "--",
            color="grey",
            label=f"m{p}",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title(f"t = {system.t[frame]:.2f}")
    plt.savefig(f"{folder_path}/{frame:03d}.png")
    plt.close(fig)
print(f"Saved frames to {folder_path}")

# list of filenames
png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# create list of image files
images = []
for png_file in png_files:
    image = Image.open(os.path.join(folder_path, png_file))
    images.append(image)

# convert to gif
images[0].save("demo.gif", save_all=True, append_images=images[1:], duration=0, loop=0)
print("Generated gif")
