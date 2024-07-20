import cv2
import numpy as np

def add_glare(image, intensity=1, radius=100, ksize=21):

    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=radius)

    blurred_image = cv2.addWeighted(image, 1, blurred_image, intensity, 0)

    return blurred_image



if __name__ == "__main__":
    # Path to your image
    image_path = './data/VW316 7CS-RCL.TLB-20220810.HV_001.png'

    # Apply glare effect
    result_image = add_glare(image_path)

    # Save or display the result
    cv2.imshow('Glare Effect', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
