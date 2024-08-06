import numpy as np
import cv2 as cv
from RawImage import RawImage


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
    

'''
`adapt_luminance` is a function that connects the algorithm with the GUI. The GUI is supposed to allow users to upload an image and enter a sample luminance value so that the image can be processed using gamma correction. The arguments `image_path` and `sample_luminance` should be extracted from users' input.
'''
def adapt_luminance(image, sample_luminance):
    # original_img = cv.imread(image_path)
    #
    # image = RawImage()
    # image.loadRGB(image_path)
    original_img = np.array(image)

    lab_original = cv.cvtColor(original_img, cv.COLOR_BGR2LAB)
    l, a, b_ch = cv.split(lab_original)

    adjusted_luminance = gamma_function(l, 0.65, 0.63, 0.5, sample_luminance**2)

    min_val = adjusted_luminance.min()
    max_val = adjusted_luminance.max()
    adjusted_luminance = ((adjusted_luminance - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    adjusted_luminance = np.clip(adjusted_luminance, 0, 255)

    adjusted_luminance = cv.resize(adjusted_luminance, (l.shape[1], l.shape[0]))

    lab_adjusted = cv.merge([adjusted_luminance, a, b_ch])

    adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)

    # comparison_img = np.hstack((original_img, adjusted_img))

    title = f"Sample Luminance: {sample_luminance:.6g} cd/m^2"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2
    color = (255, 255, 255)  # White color for text
    # adjusted_img = cv.putText(adjusted_img, title, (40, 80), font, font_scale, color, thickness, cv.LINE_AA)

    return adjusted_img