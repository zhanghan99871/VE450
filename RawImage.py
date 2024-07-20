import numpy as np
import cv2 as cv
from skimage import color
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
class RawImage:
    def __init__(self):
        self.res = [0,0]
        self.luminance = None
        self.rgb = None

    def loadRGB(self, png_file):
        img = cv.imread(png_file)
        height, width, _ = img.shape
        self.res = [height, width]
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
    def gamut_clipping(self, mode, name):
        display = {"red": [0.641, 0.332], "green": [0.297, 0.622], "blue": [0.150, 0.060]}
        to_mode_index = {"Gamut clipping": 0, "Maintain lightness and hue":1, "Maintain hue":2}
        step = 0.01
        def is_point_in_triangle(p, a, b, c):
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

            d1 = sign(p, a, b)
            d2 = sign(p, b, c)
            d3 = sign(p, c, a)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            return not (has_neg and has_pos)
        def is_in_display_range(h,l,s):
            r, g, b = colorsys.hls_to_rgb(h,l,s)
            sum = r + g + b
            if sum == 0:
                x = 0.3127
                y = 0.3290
            else:
                x = r/sum
                y = g/sum
            return is_point_in_triangle([x,y], display["red"], display["green"], display["blue"])
        #h = 300
        #w = 800
        #hue, lit, sat = colorsys.rgb_to_hls(self.rgb[h][w][0]/float(256), self.rgb[h][w][1]/float(256), self.rgb[h][w][2]/float(256))
        def gamut_clipping(h, l, s):
            target_h = 0.0
            target_l = 0.5
            target_s = 0.0
            dh = target_h - h
            dl = target_l - l
            ds = target_s - s
            magnitude = (dh**2 + dl**2 + ds**2)**0.5
            if magnitude != 0:
                dh /= magnitude
                dl /= magnitude
                ds /= magnitude
            h += dh * step
            l += dl * step
            s += ds * step
            return h, l, s
        def maintain_lightness_and_hue(h, l, s):
            s = s - step
            if s < 0:
                s = 0
            return (h,l,s)
        def maintain_hue(h, l, s):
            if s != 0:
                k = (l-0.5)/s
                s = s - step
                if s < 0 :
                    s = 0
                l = k*s+0.5
            elif l > 0.5: 
                l = l - step
            else:
                l = l + step
            return (h,l,s)
        fun = [gamut_clipping, maintain_lightness_and_hue, maintain_hue]
        #h = 15
        #w = 619
        #print(self.rgb[h][w])
        #hue, lit, sat = colorsys.rgb_to_hls(self.rgb[h][w][0]/float(256), self.rgb[h][w][1]/float(256), self.rgb[h][w][2]/float(256))
        #while not (is_in_display_range(hue, lit, sat)):
        #    print(hue, lit, sat)
        #    hue, lit, sat = fun[to_mode_index[mode]](hue,lit,sat)
        for h in range(self.res[0]):
            for w in range(self.res[1]):
                    hue, lit, sat = colorsys.rgb_to_hls(self.rgb[h][w][0]/float(256), self.rgb[h][w][1]/float(256), self.rgb[h][w][2]/float(256))
                    while not (is_in_display_range(hue, lit, sat)):
                        hue, lit, sat = fun[to_mode_index[mode]](hue,lit,sat)
                    r,g,b = colorsys.hls_to_rgb(hue,lit,sat)
                    self.rgb[h][w]=[r*256,g*256,b*256]
        im = Image.fromarray(self.rgb,"RGB")
        im.save(name)
# Example usage
# image = RawImage()
# image.loadRGB('/home/yaqing/ve450/Human_eye-Adaptation-Rendering-Algorithm/data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_init.png')
# image.convert_rgb_to_lab_luminance()
# image.saveLuminance('luminance_image.png')
