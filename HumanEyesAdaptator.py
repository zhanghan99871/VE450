import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
from RawImage import RawImage
from generate_luminance_values import LuminanceGenerator

class HumanEyesAdaptator:
    def __init__(self, initial_png_file, adjusted_png_files, initial_luminance, fit_func):
        self.X = self.extract_luminance_from_png(initial_png_file)
        self.Y_files = adjusted_png_files
        self.X_Max = self.X.max()
        self.initial_luminance = initial_luminance 
        self.luminance_generator = LuminanceGenerator(self.initial_luminance)
        self.X_Ave_values = self.generate_sample_luminance_values()
        self.fit_func = fit_func

    def extract_luminance_from_png(self, png_file):
        image = RawImage()  # txt_file is not needed here
        image.loadRGB(png_file)
        image.convert_rgb_to_lab_luminance()
        return image.luminance

    def apply_brightness_adjustment(self, df, X_Ave, k, b, c, epsilon=1e-10):
        df = df + epsilon  # Add epsilon to avoid log10(0)
        # Y = 100 * np.log10(1 + 9 * (df / self.X_Max) ** (k * np.log10(df * X_Ave) + b))
        Y = 100 * np.log10(1 + 9 * (df / self.X_Max) ** (k * np.log10(1 + c * X_Ave) + b))
        return Y

    def gamma_function(self, X, k, b, c, X_Ave):
        epsilon = 1e-10
        X = X + epsilon  # Add epsilon to avoid log10(0)
        # return 100 * np.log10(1 + 9 * (X / self.X_Max) ** (k * np.log10(X * X_Ave) + b))
        return 100 * np.log10(1 + 9 * (X / self.X_Max) ** (k * np.log10(1 + c * X_Ave) + b))
    
    def sigmoid(self, X, k, X_Ave):
        return 1 / (1 + np.exp(-k * (X - X_Ave)))

    def fit(self):
        if self.fit_func == "gamma":  # Fit gamma
            k_values = []
            b_values = []
            c_values = []
            r2_scores = []
            delta_Es = []

            for i, y_file in enumerate(self.Y_files):
                Y = self.extract_luminance_from_png(y_file)
                X_Ave = self.X_Ave_values[i]
                
                params, _ = curve_fit(lambda X, k, b, c: self.gamma_function(X, k, b, c, X_Ave), self.X.ravel(), Y.ravel())
                k_values.append(params[0])
                b_values.append(params[1])
                c_values.append(params[2])
                
                # Calculate R²
                Y_pred = self.gamma_function(self.X, params[0], params[1], params[2], X_Ave)
                r2 = r2_score(Y.ravel(), Y_pred.ravel())
                r2_scores.append(r2)

                # Calculate ΔE
                delta_E = np.sqrt(mse(Y, Y_pred))
                delta_Es.append(delta_E)
            
            k_avg = np.mean(k_values)
            b_avg = np.mean(b_values)
            c_avg = np.mean(c_values)
            r2_avg = np.mean(r2_scores)

            print(f"Fitted parameters: k = {k_avg}, b = {b_avg}, c = {c_avg}")
            print(f"Average R²: {r2_avg}")
            return k_avg, b_avg, c_avg, r2_avg, r2_scores, delta_Es
        
        elif self.fit_func == "sigmoid":  # Fit sigmoid
            k_values = []
            r2_scores = []
            delta_Es = []

            for i, y_file in enumerate(self.Y_files):
                Y = self.extract_luminance_from_png(y_file)
                X_Ave = self.X_Ave_values[i]
                
                params, _ = curve_fit(lambda X, k: self.sigmoid(X, k, X_Ave), self.X.ravel(), Y.ravel())
                k_values.append(params[0])
                
                # Calculate R²
                Y_pred = self.sigmoid(self.X, params[0], X_Ave)
                r2 = r2_score(Y.ravel(), Y_pred.ravel())
                r2_scores.append(r2)

                # Calculate ΔE
                delta_E = np.sqrt(mse(Y, Y_pred))
                delta_Es.append(delta_E)
            
            k_avg = np.mean(k_values)
            r2_avg = np.mean(r2_scores)

            print(f"Fitted parameters: k = {k_avg}")
            print(f"Average R²: {r2_avg}")
            return k_avg, r2_avg, r2_scores, delta_Es

    def generate_sample_luminance_values(self):
        return self.luminance_generator.generate_sample_luminance_values()

    def save_comparison_images(self, output_dir, k, b, c, sample_luminance_values, r2_scores, delta_Es):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, (y_file, luminance_value, r2_score_val, delta_E_val) in enumerate(zip(self.Y_files, sample_luminance_values, r2_scores, delta_Es)):
            try:
                original_img = cv.imread(y_file)
                image = RawImage()
                image.loadRGB(y_file)
                
                # Convert original image to LAB color space
                lab_original = cv.cvtColor(original_img, cv.COLOR_BGR2LAB)
                l, a, b_ch = cv.split(lab_original)
                
                # Apply brightness adjustment to the L channel
                adjusted_luminance = self.apply_brightness_adjustment(l, self.X_Ave_values[i], k, b, c)
                
                # Normalize adjusted luminance to 0-255 range
                adjusted_luminance = ((adjusted_luminance - adjusted_luminance.min()) / 
                                      (adjusted_luminance.max() - adjusted_luminance.min()) * 255).astype(np.uint8)
                
                # Merge the adjusted L channel back with the original a and b channels
                lab_adjusted = cv.merge([adjusted_luminance, a, b_ch])
                
                # Convert back to BGR color space
                adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)
                
                # Combine images side by side
                comparison_img = np.hstack((original_img, adjusted_img))
                
                # Add title with the luminance value, R² score, and ΔE
                title = f"Sample Luminance: {luminance_value:.2f} cd/m^2, R^2: {r2_score_val:.2f}, DeltaE: {delta_E_val:.2f}"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (255, 255, 255)  # White color for text
                comparison_img = cv.putText(comparison_img, title, (10, 30), font, font_scale, color, thickness, cv.LINE_AA)
                
                output_path = os.path.join(output_dir, f"comparison_{i+1}.png")
                cv.imwrite(output_path, comparison_img)
                print(f"Comparison image saved to {output_path}")
            except Exception as e:
                print(f"Error processing file {y_file}: {e}")

    def visualize_fit(self, k, b, c, r2_scores, delta_Es, output_file):
        plt.figure(figsize=(10, 6))
        
        # Plot R² values
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o')
        plt.xlabel('Image Index')
        plt.ylabel('R² Score')
        plt.title('R² Scores for Each Adjusted Image')
        
        # Plot ΔE values
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(delta_Es) + 1), delta_Es, marker='o')
        plt.xlabel('Image Index')
        plt.ylabel('ΔE')
        plt.title('ΔE for Each Adjusted Image')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Figure saved to {output_file}")

current_path = os.path.abspath(os.path.dirname(__file__))

initial_png_file = os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_original.png')
adjusted_png_files = [
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_002.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_003.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_004.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_005.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_006.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_007.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_008.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_009.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_0010.png'),
    os.path.join(current_path, 'data/VW216.RTSL-BUL.HV/VW216.RTSL-BUL.HV_0011.png'),
]
initial_luminance = 6809.47  # Initial luminance in cd/m²

adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, initial_luminance, "gamma")

# Fit gamma
k, b, c, r2_avg, r2_scores, delta_Es = adaptator.fit()

# Save comparison images
output_dir = os.path.join(current_path, 'data/comparison_images')
sample_luminance_values = adaptator.generate_sample_luminance_values()
adaptator.save_comparison_images(output_dir, k, b, c, sample_luminance_values, r2_scores, delta_Es)

# Visualize the fit and save the figure
output_file = os.path.join(current_path, 'data/r2_and_delta_e.png')
adaptator.visualize_fit(k, b, c, r2_scores, delta_Es, output_file)
