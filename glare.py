import cv2
import numpy as np

def add_glare(image, intensity=1, radius=100, ksize=21):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set your threshold value
    threshold_value = 256*0.8  # Change this value based on your needs

    # Apply the threshold
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    mask = np.stack([thresholded_image] * image.shape[-1], axis=-1)
    result_image = np.where(mask == 0, 0, image)

    blurred_image = cv2.GaussianBlur(result_image, (ksize, ksize), sigmaX=radius)

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
