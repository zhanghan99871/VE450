import cv2
import numpy as np

# Load your image (make sure to provide the correct path)
image = cv2.imread('data/VW216/VW216.RTSL-BUL.HV_001.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set your threshold value
threshold_value = 0  # Change this value based on your needs

# Apply the threshold
_, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
mask = np.stack([thresholded_image]*3, axis=-1)
print(mask.shape)

# Where thresholded_image is 0, set to black, else keep the original gray values
result_image = np.where(mask == 0, 0, image)

# Save or display the result
cv2.imwrite('output_image.jpg', result_image)
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
