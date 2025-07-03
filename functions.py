import os
import re
import matplotlib.pyplot as plt
import scipy.misc
import openslide
from openslide import open_slide, ImageSlide


def extract_patch_by_location(filepath, location, patch_size=(500, 500),
                              plot_image=False, level_to_analyze=0,
                              save=False, save_path="."):
    if not os.path.isfile(filepath):
        raise IOError("File not found: " + filepath)
        return []
    
    slide = open_slide(filepath)
    filename = re.search("(?<=/)[^/]+\.svs", filepath).group(0)[0:-4]
    slide_image = slide.read_region(location, level_to_analyze, patch_size)
    if plot_image:
        plt.figure()
        plt.imshow(slide_image)
        plt.show()
    if save:
        savename = os.path.join(
            save_path, str(filename) + "_" + str(location[0]) + "_" + str(location[1]) + ".png"
        )
        misc.imsave(savename, slide_image)
        print("Saved patch to " + savename)
    
    return slide_image