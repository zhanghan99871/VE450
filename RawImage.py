import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
class RawImage:
    def __init__(self, txt_file):
        self.luminance = None
        self.rgb = None
        self.x_range = None
        self.y_range = None
        self.img_size = None
        self.read_initial_values(txt_file)

    def read_initial_values(self, txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            self.x_range = [float(val) for val in lines[4].strip().split()[0:2]]
            self.y_range = [float(val) for val in lines[4].strip().split()[2:4]]
            self.img_size = [int(val) for val in lines[5].strip().split()[0:2]]
            self.luminance = np.zeros((self.img_size[1], self.img_size[0], 1))
            self.rgb = np.zeros((self.img_size[1], self.img_size[0], 3))

    def loadRGB(self, png_file):
        img = cv.imread(png_file)
        self.rgb = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    def loadLuminance(self, txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            start_reading = False
            for line in lines:
                if start_reading:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        try:
                            x = float(parts[0]) - self.x_range[0]
                            y = float(parts[1]) - self.y_range[0]
                            value = float(parts[2])
                            self.luminance[int(y / 0.02)][int(x / 0.02)][0] = value
                        except ValueError:
                            continue
                if 'X' in line:
                    start_reading = True

    # for check the reading
    def saveLuminance(self, file_name):
        if self.luminance is not None:
            plt.imshow(self.luminance[:, :, 0], cmap='gray')
            plt.colorbar()
            plt.savefig(file_name)
            print(f"Luminance data has been saved to {file_name}")
        else:
            print("Luminance data is not loaded.")

# Example usage
txt_file_path = '/path_to_txt_file.txt'
image = RawImage(txt_file_path)
image.loadRGB('/path_to_rgb_image.png')
image.loadLuminance(txt_file_path)

# For check the reading 
output_file_name = 'luminance_image.png'
image.saveLuminance(output_file_name)