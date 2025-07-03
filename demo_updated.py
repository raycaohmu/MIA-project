# Reconstruct segmentation
import skimage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Updated color dictionary with hex colors
color_dict = {
    "Connective": "#235cec",
    "Dead": "#feff00", 
    "Epithelial": "#ff9f44",
    "Inflammatory": "#22dd4d",
    "Neoplastic": "#ff0000",
}

# Convert hex colors to RGB for image processing
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

# Create RGB color dictionary
color_dict_rgb = {k: hex_to_rgb(v) for k, v in color_dict.items()}

h, w = 1024, 1024
# Assuming image and cell_feat_patch are already defined
image = np.array(image)[..., :3]
roi_pred_image = image.copy()

for index, (_, cell) in enumerate(cell_feat_patch.iterrows()):
    y, x = skimage.draw.ellipse(cell['coordinate_x'] - cell['coordinate_x_patch'], 
                                cell['coordinate_y'] - cell['coordinate_y_patch'], 
                                cell['MajorAxis']/2, cell['MinorAxis']/2, 
                                rotation=-cell['Rotation'])
    c = np.logical_and.reduce([x>0, x<h, y>0, y<w])
    x, y = x[c], y[c]
    roi_pred_image[x, y, :] = color_dict_rgb[cell['CellType']]

# Create the plot with legend
plt.figure(figsize=(10, 8))
plt.imshow(roi_pred_image)

# Create legend
legend_elements = [Patch(facecolor=color, label=cell_type) 
                  for cell_type, color in color_dict.items()]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.title('Cell Segmentation with Cell Types')
plt.axis('off')  # Remove axes for cleaner look
plt.tight_layout()
plt.show() 