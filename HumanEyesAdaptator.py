import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
import logging
from RawImage import RawImage
from generate_luminance_values import LuminanceGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def apply_brightness_adjustment(self, df, X_Ave, k, b, c, epsilon=1e-10, max_val=1e3):
        try:
            df = df + epsilon  # Add epsilon to avoid log10(0)
            X_Ave_adjusted = 1 + c * X_Ave
            X_Ave_adjusted = np.clip(X_Ave_adjusted, epsilon, max_val)
            
            inner_term = df / self.X_Max
            inner_term = np.clip(inner_term, epsilon, max_val)
            
            exponent = k * np.log10(np.clip(X_Ave_adjusted, epsilon, max_val)) + b
            exponent = np.clip(exponent, -max_val, max_val)
            
            with np.errstate(over='ignore'):  # Suppress overflow warnings
                power_term = np.power(inner_term, exponent)
                power_term = np.clip(power_term, -max_val, max_val)
            
            Y = 100 * np.log10(1 + 9 * power_term)
            logging.info(f"Brightness adjustment applied successfully.")
            return Y
        except Exception as e:
            logging.error(f"Error in apply_brightness_adjustment: {e}")
            return np.zeros_like(df)  # Return a default value to prevent crashes

    def gamma_function(self, X, k, b, c, X_Ave, epsilon=1e-10, max_val=1e3):
        X = X + epsilon  # Add epsilon to avoid log10(0)
        X_Ave_adjusted = 1 + c * X_Ave
        if isinstance(X_Ave_adjusted, np.ndarray):
            X_Ave_adjusted[X_Ave_adjusted <= 0] = epsilon  # Ensure positive values
        else:
            X_Ave_adjusted = max(X_Ave_adjusted, epsilon)
        
        inner_term = X / self.X_Max
        inner_term[inner_term <= 0] = epsilon  # Ensure positive values
        inner_term = np.clip(inner_term, 0, max_val)  # Cap to max_val

        exponent = k * np.log10(np.clip(X_Ave_adjusted, epsilon, max_val)) + b
        exponent = np.clip(exponent, -max_val, max_val)  # Cap to prevent overflow

        with np.errstate(over='ignore'):  # Suppress overflow warnings
            power_term = np.power(inner_term, exponent)
            power_term = np.clip(power_term, -max_val, max_val)  # Cap to prevent overflow

        gamma_result = 100 * np.log10(1 + 9 * power_term)
        return gamma_result
    
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
                
                # Provide initial guesses and bounds for parameters
                initial_guesses = [10.0, 0.0, 5.0]
                bounds = ([1, -10, 1], [50, 50, 50])
                
                try:
                    params, _ = curve_fit(
                        lambda X, k, b, c: self.gamma_function(X, k, b, c, X_Ave),
                        self.X.ravel(), Y.ravel(),
                        p0=initial_guesses, bounds=bounds
                    )
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
                except RuntimeError as e:
                    print(f"Fit did not converge for file {y_file}: {e}")
                    continue
                
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
                
                # Provide initial guesses and bounds for parameters
                initial_guesses = [1.0]
                bounds = ([0], [np.inf])
                
                try:
                    params, _ = curve_fit(
                        lambda X, k: self.sigmoid(X, k, X_Ave),
                        self.X.ravel(), Y.ravel(),
                        p0=initial_guesses, bounds=bounds
                    )
                    k_values.append(params[0])
                    
                    # Calculate R²
                    Y_pred = self.sigmoid(self.X, params[0], X_Ave)
                    r2 = r2_score(Y.ravel(), Y_pred.ravel())
                    r2_scores.append(r2)

                    # Calculate ΔE
                    delta_E = np.sqrt(mse(Y, Y_pred))
                    delta_Es.append(delta_E)
                except RuntimeError as e:
                    print(f"Fit did not converge for file {y_file}: {e}")
                    continue
                
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
                if original_img is None:
                    logging.error(f"Failed to load image file {y_file}")
                    continue

                image = RawImage()
                image.loadRGB(y_file)
                
                # Convert original image to LAB color space
                lab_original = cv.cvtColor(original_img, cv.COLOR_BGR2LAB)
                l, a, b_ch = cv.split(lab_original)

                # Ensure that the original L channel is in the correct range
                logging.info(f"Original L channel min: {l.min()}, max: {l.max()}")
                
                # Apply brightness adjustment to the L channel
                adjusted_luminance = self.apply_brightness_adjustment(l, self.X_Ave_values[i], k, b, c)
                
                # Log the min and max values of adjusted luminance before normalization
                logging.info(f"Adjusted luminance min: {adjusted_luminance.min()}, max: {adjusted_luminance.max()}")
                
                # Normalize adjusted luminance to 0-255 range
                min_val = adjusted_luminance.min()
                max_val = adjusted_luminance.max()
                if max_val - min_val == 0:
                    logging.error(f"Normalization error: max_val ({max_val}) - min_val ({min_val}) = 0")
                    continue
                
                # Check if min_val is non-zero and log a warning
                if min_val != 0:
                    logging.warning(f"Adjusted luminance min is not zero: {min_val}")

                adjusted_luminance = ((adjusted_luminance - min_val) / 
                                    (max_val - min_val) * 255).astype(np.uint8)
                
                # Log the min and max values of adjusted luminance after normalization
                logging.info(f"Normalized luminance min: {adjusted_luminance.min()}, max: {adjusted_luminance.max()}")
                
                # Ensure luminance values are correctly processed
                adjusted_luminance = np.clip(adjusted_luminance, 0, 255)

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
                logging.info(f"Comparison image saved to {output_path}")
            except Exception as e:
                logging.error(f"Error processing file {y_file}: {e}")

    def visualize_fit(self, k, b, c, r2_scores, delta_Es, output_file, r2_avg):
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
        
        # Add k, b, c values and average R² to the plot
        plt.figtext(0.5, 0.01, f'Fitted parameters: k = {k:.2f}, b = {b:.2f}, c = {c:.2f} | Average R²: {r2_avg:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        logging.info(f"Figure saved to {output_file}")

current_path = os.path.abspath(os.path.dirname(__file__))

# TODO: change with real path
initial_png_file = os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_2654.74.png')
adjusted_png_files = [
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_001.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_002.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_003.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_004.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_005.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_006.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_007.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_008.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_009.png'),
    os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_0010.png'),
]
initial_luminance = 2654.74  # TODO: CHANGE Initial luminance in cd/m²

adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, initial_luminance, "gamma")

# Fit gamma
k, b, c, r2_avg, r2_scores, delta_Es = adaptator.fit()

# TODO: change with real path, save comparison images
output_dir = os.path.join(current_path, 'data/comparison_images_vw310')
sample_luminance_values = adaptator.generate_sample_luminance_values()
adaptator.save_comparison_images(output_dir, k, b, c, sample_luminance_values, r2_scores, delta_Es)

# Visualize the fit and save the figure
output_file = os.path.join(output_dir, 'r2_and_delta_e.png')
adaptator.visualize_fit(k, b, c, r2_scores, delta_Es, output_file, r2_avg)
