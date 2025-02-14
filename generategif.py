import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from pirke import Pirke

# Function to generate a stack of images
def generate_images(num_images=10, resolution=128):
    images_up = []
    images_down = []
    
    shift_min_max = 0
    random_rotate = False
    canvas_range = (-3.5 - shift_min_max, 3.5 + shift_min_max)

    for _ in range(num_images):
        # Create figure
        figure = Pirke(canvas_range=canvas_range)

        # Sample random pose
        g = [
            np.random.uniform(-shift_min_max, shift_min_max),
            np.random.uniform(-shift_min_max, shift_min_max),
            np.random.uniform(0, 2 * np.pi) if random_rotate else 0.0
        ]

        # Generate figure with arms up
        figure.resample(g=g, pose_class='up')
        image_up, _, _ = figure.render(image_size=resolution)
        images_up.append(Image.fromarray(image_up))

        # Generate figure with arms down
        figure.resample(g=g, pose_class='down')
        image_down, _, _ = figure.render(image_size=resolution)
        images_down.append(Image.fromarray(image_down))

    return images_up, images_down

# Function to create an animated GIF
def save_gif(images, filename, duration=200):
    images[0].save(
        filename, save_all=True, append_images=images[1:], duration=duration, loop=0
    )

# Generate images
num_images = 20
images_up, images_down = generate_images(num_images=num_images, resolution=256)

# Save animated GIFs
# save_gif(images_up, "stick_figure_arms_up.gif")
# save_gif(images_down, "stick_figure_arms_down.gif")
save_gif(images_up + images_down, "stick_figure_arms_up_and_down_static.gif")

print("GIFs saved successfully!")
