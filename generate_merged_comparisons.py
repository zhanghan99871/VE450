import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from RawImage import RawImage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gamma_function(X, k, b, c, X_Ave, epsilon=1e-10, max_val=1e3):
    X = X + epsilon  # Add epsilon to avoid log10(0)
    X_Ave_adjusted = 1 + c * X_Ave
    if isinstance(X_Ave_adjusted, np.ndarray):
        X_Ave_adjusted[X_Ave_adjusted <= 0] = epsilon  # Ensure positive values
    else:
        X_Ave_adjusted = max(X_Ave_adjusted, epsilon)
    
    inner_term = X / X.max()
    inner_term[inner_term <= 0] = epsilon  # Ensure positive values
    inner_term = np.clip(inner_term, 0, max_val)  # Cap to max_val

    exponent = k * np.log10(np.clip(X_Ave_adjusted, epsilon, max_val)) + b
    exponent = np.clip(exponent, -max_val, max_val)  # Cap to prevent overflow

    with np.errstate(over='ignore'):  # Suppress overflow warnings
        power_term = np.power(inner_term, exponent)
        power_term = np.clip(power_term, -max_val, max_val)  # Cap to prevent overflow

    gamma_result = 100 * np.log10(1 + 9 * power_term)
    return gamma_result

if not os.path.exists('data/my_comparison_images'):
    os.makedirs('data/my_comparison_images')

pic_path = 'data/myPic'
y_files = os.listdir('data/myPic')

for y_file in y_files:
    name = y_file.split('.')[0]
    
    if not os.path.exists(f'data/my_comparison_images/{name}'):
        os.makedirs(f'data/my_comparison_images/{name}')
        
    original_img = cv.imread(f'{pic_path}/{y_file}')
    
    image = RawImage()
    image.loadRGB(f'{pic_path}/{y_file}')
    
    # Convert original image to LAB color space
    lab_original = cv.cvtColor(original_img, cv.COLOR_BGR2LAB)
    l, a, b_ch = cv.split(lab_original)
    
    # Ensure that the original L channel is in the correct range
    logging.info(f"Original L channel min: {l.min()}, max: {l.max()}")
    
    sample_luminance_values = [10, 100, 1000, 10000, 100000]
    sample_luminance_values.sort()  # Sort the luminance values

    comparison_images = [original_img]

    for i, luminance_value in enumerate(sample_luminance_values):
        adjusted_luminance = gamma_function(l, 0.65, 0.63, 0.5, luminance_value)
        
        # Log the min and max values of adjusted luminance before normalization
        logging.info(f"Adjusted luminance min: {adjusted_luminance.min()}, max: {adjusted_luminance.max()}")
        
        # Normalize adjusted luminance to 0-255 range
        min_val = adjusted_luminance.min()
        max_val = adjusted_luminance.max()
        adjusted_luminance = ((adjusted_luminance - min_val) / 
                            (max_val - min_val) * 255).astype(np.uint8)
        logging.info(f"Normalized luminance min: {adjusted_luminance.min()}, max: {adjusted_luminance.max()}")
        # Ensure luminance values are correctly processed
        adjusted_luminance = np.clip(adjusted_luminance, 0, 255)
        # Ensure the adjusted luminance has the same shape as the original channels
        adjusted_luminance = cv.resize(adjusted_luminance, (l.shape[1], l.shape[0]))
        # Merge the adjusted L channel back with the original a and b channels
        lab_adjusted = cv.merge([adjusted_luminance, a, b_ch])
        # Convert back to BGR color space
        adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)
        
        # Add title with luminance value
        title = f"Environment Luminance: {luminance_value:.2f} cd/m^2"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        color = (0, 0, 255)  # Red color for text
        adjusted_img = cv.putText(adjusted_img, title, (10, 100), font, font_scale, color, thickness, cv.LINE_AA)
        
        comparison_images.append(adjusted_img)
    
    # Split the images into two rows
    half = len(comparison_images) // 2
    row1 = np.hstack(comparison_images[:half])
    row2 = np.hstack(comparison_images[half:])
    final_comparison_image = np.vstack((row1, row2))
    
    output_path = f'data/my_comparison_images/{name}/my_comparison.jpg'
    cv.imwrite(output_path, final_comparison_image)
    logging.info(f"Final comparison image saved to {output_path}")
