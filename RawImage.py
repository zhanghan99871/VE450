import numpy as np
import cv2 as cv
from PIL import Image


class RawImage:
    def __init__(self, x, y):
        self.luminance = np.zeros((x, y, 1))
        self.rgb = np.zeros((x, y, 3))

    def loadRGB(png_file):
        # TODO:
        pass

    def loadLuminance(txt_file):
        # TODO:
        pass

