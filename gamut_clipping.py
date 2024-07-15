import RawImage
import math
import colorsys

def eval(img1, img2):
    total_sum = 0
    for h in range(img1.res[0]):
        for w in range(img1.res[1]):
            diff = [0.0,0.0,0.0]
            hue,lit,sat = colorsys.rgb_to_hls(img1.rgb[h][w][0]/float(256), img1.rgb[h][w][1]/float(256), img1.rgb[h][w][2]/float(256))
            img1_hls = [hue,lit,sat]
            hue,lit,sat = colorsys.rgb_to_hls(img2.rgb[h][w][0]/float(256), img2.rgb[h][w][1]/float(256), img2.rgb[h][w][2]/float(256))
            img2_hls = [hue,lit,sat]
            for i in range(3):
                dif = img1_hls[i] - img2_hls[i]
                diff[i] = dif*dif
            sum = 0.0
            for i in range(3):
                sum += diff[i]
            sum = math.sqrt(sum)
            total_sum += sum
    return total_sum