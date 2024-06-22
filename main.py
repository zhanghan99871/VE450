import RawImage


# Replace with actual file path
def main():
    img = RawImage.RawImage(600, 1500)
    img.loadRGB("./data/VW216.RTSL-BUL.HV_001.png")
    img.loadLuminance("./data/VW216.RTSL-BUL.HV_001.txt")
    r_list = []
    g_list = []
    b_list = []
    l_list = []
    for i in range(600):
        for j in range(1500):
            r_list.append(img.rgb[i][j][0])
            g_list.append(img.rgb[i][j][1])
            b_list.append(img.rgb[i][j][2])
            l_list.append(img.luminance[i][j])
    print(max(r_list))
    print(min(r_list))
    print(max(g_list))
    print(min(g_list))
    print(max(b_list))
    print(min(b_list))
    print(max(l_list))
    print(min(l_list))


if __name__ == "__main__":
    main()

