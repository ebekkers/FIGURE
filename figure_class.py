import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

class Shape:
    def __init__(self, 
                 g0_base=(0, 0, np.pi / 2), 
                 w_range=(0.8, 1.2), 
                 h_range=(1.8, 2.2), 
                 l_arm_range=(0.7, 1.0), 
                 l_leg_range=(0.9, 1.2), 
                 torso_colors=["red", "green", "blue", "gold"],
                 color_probabilities={"up-up": [1.0, 0.0, 0.0, 0.0], 
                                      "down-down": [0.0, 1.0, 0.0, 0.0],
                                      "down-down": [0.0, 0.0, 1.0, 0.0],
                                      "down-down": [0.0, 0.0, 0.0, 1.0]},
                 color_consistency=1.0,
                 canvas_range=(-5.5,5.5)):
        
        self.g0_base = np.array(g0_base)  # Initial pose (x, y, theta)
        
        # Sample body proportions from given ranges
        self.w_range = w_range
        self.h_range = h_range
        self.l_arm_range = l_arm_range
        self.l_leg_range = l_leg_range        
        self.torso_colors = torso_colors
        self.color_probabilities = color_probabilities
        for key in self.color_probabilities:
            self.color_probabilities[key] = np.array(color_probabilities[key]) * color_consistency + (1 - np.array(color_probabilities[key])) * (1 - color_consistency) / (len(color_probabilities[key])-1)
        self.canvas_range = canvas_range
        self.joint_var = 20
        
        # Define the possible arm classes
        self.arm_classes = ["up-up", "down-down", "up-down", "down-up"]
        
        self.resample(g=[0.,0.,0.], pose_class = 'up-up')

    def set_g0(self, g):
        self.g = g
        self.g0 = self.transform(g, self.g0_base[:2], self.g0_base[2])

    def set_pose_class(self, pose_class):
        self.pose_class = pose_class
    
    def sample_body_proportions(self):
        self.w = np.random.uniform(*self.w_range)
        self.h = np.random.uniform(*self.h_range)
        self.l_arm = np.random.uniform(*self.l_arm_range)
        self.l_leg = np.random.uniform(*self.l_leg_range)

    def sample_angles(self, pose_class):
        """Samples joint angles based on the pose class."""

        if pose_class == "up-up":
            theta1 = np.radians(np.random.uniform(-85, -10))
            theta3 = np.radians(np.random.uniform(10, 85))
        elif pose_class == "down-down":
            theta1 = np.radians(np.random.uniform(-170, -95))
            theta3 = np.radians(np.random.uniform(95, 170))
        elif pose_class == "up-down":
            theta1 = np.radians(np.random.uniform(-85, -10))
            theta3 = np.radians(np.random.uniform(95, 170))
        elif pose_class == "down-up":
            theta1 = np.radians(np.random.uniform(-170, -95))
            theta3 = np.radians(np.random.uniform(10, 85))
        else:
            raise ValueError(f"Unknown class label: {pose_class}")
               
        angles = {
            "theta1": theta1,
            "theta2": np.radians(np.random.uniform(-self.joint_var, self.joint_var)),
            "theta3": theta3,
            "theta4": np.radians(np.random.uniform(-self.joint_var, self.joint_var)),
            "theta5": np.radians(np.random.uniform(170, 225)),
            "theta6": np.radians(np.random.uniform(-self.joint_var, self.joint_var)),
            "theta7": np.radians(np.random.uniform(135, 190)),
            "theta8": np.radians(np.random.uniform(-self.joint_var, self.joint_var)),
        }
        self.angles = angles
    
    def sample_torso_color(self, pose_class):
        """Samples torso color based on the defined probability distribution."""
        probabilities = self.color_probabilities.get(pose_class, [1/len(self.torso_colors)] * len(self.torso_colors))
        self.torso_color = np.random.choice(self.torso_colors, p=probabilities)

    def transform(self, parent, offset, theta):
        """Apply SE(2) transformation hierarchically."""
        px, py, ptheta = parent
        R = np.array([[np.cos(ptheta), -np.sin(ptheta)],
                        [np.sin(ptheta), np.cos(ptheta)]])
        new_offset = np.dot(R, offset)  # Apply full parent rotation to offset
        new_pos = new_offset + [px, py]
        return np.array([new_pos[0], new_pos[1], ptheta + theta])

    def compute_kinematics(self, g):
        """Computes SE(2) elements, 2D limb positions, and all relevant points."""
               
        # Move the body to the specified pose g 
        points_se2 = {"g0": self.g0}
        limbs_r2 = {}
        points_r2 = {"g0": self.g0[:2]}
        
        limb_definitions = [
            ("g1", "g0", [self.h / 2, -self.w / 2], "theta1"),
            ("g2", "g1", [self.l_arm, 0], "theta2"),
            ("g3", "g0", [self.h / 2, self.w / 2], "theta3"),
            ("g4", "g3", [self.l_arm, 0], "theta4"),
            ("g5", "g0", [-self.h / 2, -self.w / 2], "theta5"),
            ("g6", "g5", [self.l_leg, 0], "theta6"),
            ("g7", "g0", [-self.h / 2, self.w / 2], "theta7"),
            ("g8", "g7", [self.l_leg, 0], "theta8")
        ]
        
        for child, parent, offset, theta_key in limb_definitions:
            points_se2[child] = self.transform(points_se2[parent], np.array(offset), self.angles[theta_key])
            points_r2[child] = points_se2[child][:2]
            
            # Compute tilde points for end of limbs
            limb_length = self.l_arm if child in ["g1", "g2", "g3", "g4"] else self.l_leg
            tilde_offset = np.array([limb_length, 0])
            points_r2[f"{child}_tilde"] = self.transform(points_se2[child], tilde_offset, 0)[:2]
            limbs_r2[child] = (points_r2[child], points_r2[f"{child}_tilde"])
        
        self.points_se2 = points_se2
        self.limbs_r2 = limbs_r2
        self.points_r2 = points_r2
    
    def resample(self, g, pose_class):
        """Resamples the joint angles and updates kinematics."""
        self.set_g0(g)
        self.set_pose_class(pose_class)
        self.sample_body_proportions()
        self.sample_angles(pose_class)
        self.compute_kinematics(g)
        self.sample_torso_color(pose_class)
    
    def visualize(self):
        """Visualizes the stick figure using matplotlib."""

        fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        ax.set_xlim(self.canvas_range[0], self.canvas_range[1])
        ax.set_ylim(self.canvas_range[0], self.canvas_range[1])
        ax.set_axis_off()  # Hide all axes


        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        
        # Draw torso as a filled rectangle
        torso_center = self.points_r2["g0"]
        torso_rect = plt.Rectangle(
            torso_center - [self.w / 2, self.h / 2], 
            self.w, self.h, color=self.torso_color, alpha=0.5, angle=np.degrees(self.g[2]),
            rotation_point='center'
        )
        ax.add_patch(torso_rect)
        
        # Draw limbs as lines
        for start, end in self.limbs_r2.values():
            ax.plot([start[0], end[0]], [start[1], end[1]], '-', linewidth=7, color='gray')
        
        # Draw head as a circle
        head_radius = 0.9 * self.w / 2
        head_center = self.transform(self.g0, [self.h / 2 + head_radius - 0.1, 0.0], 0.0)
        ax.add_patch(plt.Circle(head_center, head_radius, color='gray'))

        return fig
    
    def world_to_pixel(self, x, y, canvas_range, dim_i, dim_j):
        """Convert world coordinates (x, y) to pixel indices (i, j), preserving aspect ratio."""
        min_x, max_x = canvas_range
        min_y, max_y = canvas_range
        width, height = max_x - min_x, max_y - min_y
        delta_x = width / dim_j
        delta_y = height / dim_i

        j = round((x - min_x) / delta_x - 0.5)
        i = round(dim_i - (y - min_y) / delta_y + 0.5)

        i = min(max(0,i),dim_i-1)
        j = min(max(0,j),dim_j-1)

        return i, j

    def render(self, image_size=64):
        """
        Renders the stick figure into a pixelated image.
        
        Args:
            image_size: The desired output square image size (pixels).
        
        Returns:
            image_array: The rasterized grayscale image of the stick figure.
            points_r2_pixel: The points in image pixel coordinates.
            points_se2_pixel: The transformed SE(2) points in pixel coordinates.
        """
        fig = self.visualize()
        ax = fig.gca()
        ax.axis('off')
        
        # Render figure to an image buffer
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())

        # Resize image to desired dimensions
        img = Image.fromarray(image_array)
        img_resized = img.resize((image_size, image_size))
        image_array = np.array(img_resized)
        image_array = image_array[:,:,:3]  # only take rgb
        
        points_r2_pixel = []
        for g in ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g2_tilde', 'g4_tilde', 'g6_tilde', 'g8_tilde']:
            x, y = self.points_r2[g]
            i, j = self.world_to_pixel(x, y, self.canvas_range, image_size, image_size)
            points_r2_pixel.append((i,j))
        points_se2_pixel = []
        for g in ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8']:
            x, y, theta = self.points_se2[g]
            i, j = self.world_to_pixel(x, y, self.canvas_range, image_size, image_size)
            angle = theta + np.pi / 2
            points_se2_pixel.append((i,j,angle))

        # just for testing
        # for (i,j) in points_r2_pixel:
            # image_array[i,j] = image_array[i,j] * 0
        # for (i,j,t) in points_se2_pixel:
            # image_array[i,j] = image_array[i,j] * 0

        plt.close(fig) 
               
        return image_array, points_r2_pixel, points_se2_pixel

# Example usage
if __name__ == "__main__":
    shift_min_max = 2
    random_rotate = False
    canvas_range = (-3.5 - shift_min_max, 3.5 + shift_min_max)
    resolution = 128
    figure = Shape(canvas_range=canvas_range)

    # Sample random pose
    g = [
            np.random.uniform(-shift_min_max,shift_min_max), 
            np.random.uniform(-shift_min_max,shift_min_max), 
            np.random.uniform(0, 2 * np.pi) if random_rotate else 0.0
        ]
    # Generate figure at that pose
    figure.resample(g=g, pose_class='up-up')
    # And render
    image, points_r2, points_se2 = figure.render(image_size=resolution)

    # Show the rendered image
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(image, aspect='auto')
    ax.imshow(image, aspect='equal')

    1+1
