import RawImage
import gamut_clipping
mode = ["Gamut clipping", "Maintain lightness and hue", "Maintain hue"]
# Replace with actual file path
def main():
    image = RawImage.RawImage()
    image.loadRGB('./data/VW216/VW216.RTSL-BUL.HV_002.png')
    speos = RawImage.RawImage()
    speos.loadRGB('./gamut clipping/VW216.RTSL-BUL.HV_001.png')
    #image.gamut_clipping(mode[0], "0.png")
    res = gamut_clipping.eval(image, speos)
    print(res)

if __name__ == "__main__":
    main()

