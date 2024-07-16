import numpy as np
import cv2 as cv
import os
from RawImage import RawImage

class ImageAdjuster:
    def __init__(self, k, b, c, X_Max):
        self.k = k
        self.b = b
        self.c = c
        self.X_Max = X_Max

    def apply_brightness_adjustment(self, df, X_Ave, epsilon=1e-10, max_val=1e3):
        try:
            df = df + epsilon  # Add epsilon to avoid log10(0)
            X_Ave_adjusted = 1 + self.c * X_Ave
            X_Ave_adjusted = np.clip(X_Ave_adjusted, epsilon, max_val)
            
            inner_term = df / self.X_Max
            inner_term = np.clip(inner_term, epsilon, max_val)
            
            exponent = self.k * np.log10(np.clip(X_Ave_adjusted, epsilon, max_val)) + self.b
            exponent = np.clip(exponent, -max_val, max_val)
            
            with np.errstate(over='ignore'):  # Suppress overflow warnings
                power_term = np.power(inner_term, exponent)
                power_term = np.clip(power_term, -max_val, max_val)
            
            Y = 100 * np.log10(1 + 9 * power_term)
            return Y
        except Exception as e:
            return np.zeros_like(df)  # Return a default value to prevent crashes

    def adjust_image(self, input_png_file, output_png_file, X_Ave):
        try:
            original_img = cv.imread(input_png_file)
            if original_img is None:
                return

            image = RawImage()
            image.loadRGB(input_png_file)
            
            # Convert original image to LAB color space
            lab_original = cv.cvtColor(original_img, cv.COLOR_BGR2LAB)
            l, a, b_ch = cv.split(lab_original)
            
            # Apply brightness adjustment to the L channel
            adjusted_luminance = self.apply_brightness_adjustment(l, X_Ave)
            
            # Normalize adjusted luminance to 0-255 range
            min_val = adjusted_luminance.min()
            max_val = adjusted_luminance.max()
            if max_val - min_val == 0:
                return
            
            adjusted_luminance = ((adjusted_luminance - min_val) / 
                                (max_val - min_val) * 255).astype(np.uint8)
            
            # Ensure luminance values are correctly processed
            adjusted_luminance = np.clip(adjusted_luminance, 0, 255)

            # Merge the adjusted L channel back with the original a and b channels
            lab_adjusted = cv.merge([adjusted_luminance, a, b_ch])
            
            # Convert back to BGR color space
            adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)
            
            # Save the adjusted image
            cv.imwrite(output_png_file, adjusted_img)
        except Exception as e:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Adjust the brightness of an image using fitted parameters.')
    parser.add_argument('input_png_file', type=str, help='Path to the input PNG file')
    parser.add_argument('output_png_file', type=str, help='Path to save the adjusted PNG file')
    parser.add_argument('--k', type=float, required=True, help='Fitted parameter k')
    parser.add_argument('--b', type=float, required=True, help='Fitted parameter b')
    parser.add_argument('--c', type=float, required=True, help='Fitted parameter c')
    parser.add_argument('--X_Max', type=float, default=255, help='Maximum luminance value from the fitting process')
    parser.add_argument('--X_Ave', type=float, required=True, help='Average luminance value used during the fitting process')

    args = parser.parse_args()

    # Initialize the ImageAdjuster with the fitted parameters
    image_adjuster = ImageAdjuster(args.k, args.b, args.c, args.X_Max)

    # Adjust the input image and save the result
    image_adjuster.adjust_image(args.input_png_file, args.output_png_file, args.X_Ave)
