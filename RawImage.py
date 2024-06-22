import numpy as np
import cv2 as cv


class RawImage:
    def __init__(self, x, y):
        self.luminance = np.zeros((x, y, 1))
        self.rgb = np.zeros((x, y, 3))

    def loadRGB(self, png_file):
        img = cv.imread(png_file)
        self.rgb = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    def loadLuminance(self, txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            # flag for reading the content below the "X Y value" line
            start_reading = False
            for line in lines:
                if start_reading:
                    parts = line.strip().split()
                    # TODO: add the correct number according to
                    # the txt file's begining lines
                    x = float(parts[0]) + 15
                    y = float(parts[1]) + 7
                    value = float(parts[2])
                    self.luminance[int(y / 0.02)][int(x / 0.02)][0] = value
                if 'X' in line:
                    start_reading = True

