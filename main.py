import numpy as np
from PIL import Image
def read_png_export(file_path):
    img = Image.open(file_path)
    img_arr = np.array(img)
    return img_arr

def read_txt_export(file_path):
    # Initialize lists to store data
    data = np.zeros((600, 1500, 1))

    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_reading = False
    for line in lines:
        if start_reading:
            parts = line.strip().split()
            x = float(parts[0]) + 15
            y = float(parts[1]) + 7
            value = float(parts[2])
            data[int(y / 0.02)][int(x / 0.02)][0] = value

        if 'X' in line:
            start_reading = True
    return data


    # # Skip initial non-data lines (up to row 9) and read data from row 10 onwards
    # for line in lines[9:]:
    #     parts = line.strip().split()
    #     if parts and all(part.replace('.', '', 1).isdigit() for part in parts):  # Check if all parts are numeric
    #         data.append([float(val) for val in parts])
    #
    # # Create a DataFrame from the data
    # arr = np.array(data)
    # return arr

# Replace with actual file path
def main():
    txt_file_path = './data/VW216.RTSL-BUL.HV_001.txt'
    png_file_path = "./data/VW216.RTSL-BUL.HV_001.png"
    luminance = read_txt_export(txt_file_path)
    rgb = read_png_export(png_file_path)
    data = np.concatenate((rgb, luminance), axis=2)
    r_list = []
    g_list = []
    b_list = []
    l_list = []
    for i in range(600):
        for j in range(1500):
            r_list.append(data[i][j][0])
            g_list.append(data[i][j][1])
            b_list.append(data[i][j][2])
            l_list.append(data[i][j][3])
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


    
# import cv2
#
# # Load an HDR image
# hdr_image = cv2.imread('./VW216.RTSL-BUL.HV_001.hdr', flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
# hdr_arr = np.array(hdr_image)
# # hdr_arr = hdr_arr * 255.0
#
# # Display the image (scaling it down for visibility purposes)
# print(hdr_arr.max())
# cv2.imshow('HDR Image', hdr_image * 255.0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

