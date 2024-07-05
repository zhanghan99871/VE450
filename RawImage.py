import numpy as np
import cv2 as cv
from skimage import color
import matplotlib.pyplot as plt

class RawImage:
    def __init__(self):
        self.luminance = None
        self.rgb = None

    def loadRGB(self, png_file):
        img = cv.imread(png_file)
        self.rgb = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    def convert_rgb_to_lab_luminance(self):
        if self.rgb is not None:
            lab = color.rgb2lab(self.rgb / 255.0)
            self.luminance = lab[:, :, 0]
        else:
            raise ValueError("RGB data is not loaded.")

    def saveLuminance(self, file_name):
        if self.luminance is not None:
            plt.imshow(self.luminance, cmap='gray')
            plt.colorbar()
            plt.savefig(file_name)
            print(f"Luminance data has been saved to {file_name}")
        else:
            print("Luminance data is not loaded.")

# Example usage
image = RawImage()
image.loadRGB('/home/yaqing/ve450/Human_eye-Adaptation-Rendering-Algorithm/data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_init.png')
image.convert_rgb_to_lab_luminance()
image.saveLuminance('luminance_image.png')
