import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
import logging
from RawImage import RawImage
from generate_luminance_values import LuminanceGenerator
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

class HumanEyesAdaptator:
    def __init__(self, initial_png_file, adjusted_png_files, initial_luminance, fit_func, luminance_file=None):
        self.X = self.extract_luminance_from_png(initial_png_file)
        self.Y_files = adjusted_png_files
        self.X_Max = self.X.max()
        self.initial_luminance = initial_luminance 
        self.luminance_generator = LuminanceGenerator(self.initial_luminance)
        self.fit_func = fit_func

        if luminance_file:
            self.X_Ave_values = self.read_luminance_from_file(luminance_file)
        else:
            self.X_Ave_values = self.generate_sample_luminance_values()

    def extract_luminance_from_png(self, png_file):
        image = RawImage()  # txt_file is not needed here
        image.loadRGB(png_file)
        image.convert_rgb_to_lab_luminance()
        return image.luminance

    def read_luminance_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                luminance_values = [float(line.split()[0]) for line in file.readlines()]
            logging.info(f"Luminance values read successfully from {file_path}")
            return luminance_values
        except Exception as e:
            logging.error(f"Error reading luminance values from file {file_path}: {e}")
            return []

    def apply_brightness_adjustment(self, df, L, k, x0, b):
        try:
            adjusted_luminance = sigmoid(df, L, k, x0, b)
            logging.info(f"Brightness adjustment applied successfully.")
            return adjusted_luminance
        except Exception as e:
            logging.error(f"Error in apply_brightness_adjustment: {e}")
            return np.zeros_like(df)  # Return a default value to prevent crashes

    def fit(self):
        L_values = []
        k_values = []
        x0_values = []
        b_values = []
        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(self.Y_files):
            Y = self.extract_luminance_from_png(y_file)
            X_Ave = self.X_Ave_values[i]

            # Provide initial guesses and bounds for parameters
            initial_guesses = [100, 1, np.median(self.X), 0.5]
            bounds = ([50, 0.1, 0, 0], [200, 10, 255, 100])
            
            try:
                params, _ = curve_fit(sigmoid, self.X.ravel(), Y.ravel(), p0=initial_guesses, bounds=bounds)
                L_values.append(params[0])
                k_values.append(params[1])
                x0_values.append(params[2])
                b_values.append(params[3])

                # Calculate R²
                Y_pred = sigmoid(self.X, *params)
                r2 = r2_score(Y.ravel(), Y_pred.ravel())
                r2_scores.append(r2)

                # Calculate ΔE
                delta_E = np.sqrt(mse(Y, Y_pred))
                delta_Es.append(delta_E)
            except RuntimeError as e:
                print(f"Fit did not converge for file {y_file}: {e}")
                continue
            
        r2_avg = np.mean(r2_scores)

        print(f"Fitted parameters: L = {np.mean(L_values)}, k = {np.mean(k_values)}, x0 = {np.mean(x0_values)}, b = {np.mean(b_values)}")
        print(f"Average R²: {r2_avg}")
        return L_values, k_values, x0_values, b_values, r2_avg, r2_scores, delta_Es

    def save_comparison_images(self, output_dir, L_values, k_values, x0_values, b_values, r2_scores, delta_Es, generalized=False):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, (y_file, r2_score_val, delta_E_val) in enumerate(zip(self.Y_files, r2_scores, delta_Es)):
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
                if generalized:
                    adjusted_luminance = self.apply_brightness_adjustment(l, L_values, k_values, x0_values, b_values)
                else:
                    adjusted_luminance = self.apply_brightness_adjustment(l, L_values[i], k_values[i], x0_values[i], b_values[i])
                
                # Log the min and max values of adjusted luminance
                logging.info(f"Adjusted luminance min: {adjusted_luminance.min()}, max: {adjusted_luminance.max()}")
                
                # Normalize adjusted luminance to 0-255 range
                min_val = adjusted_luminance.min()
                max_val = adjusted_luminance.max()
                adjusted_luminance = ((adjusted_luminance - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                # Ensure luminance values are correctly processed
                adjusted_luminance = np.clip(adjusted_luminance, 0, 255)

                # Merge the adjusted L channel back with the original a and b channels
                lab_adjusted = cv.merge([adjusted_luminance, a, b_ch])
                
                # Convert back to BGR color space
                adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)
                
                # Combine images side by side
                comparison_img = np.hstack((original_img, adjusted_img))
                
                # Add title with the R² score and ΔE
                title = f"R^2: {r2_score_val:.2f}, DeltaE: {delta_E_val:.2f}"
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

def fit_on_all_data_sets(data_sets, fit_func, output_base_dir):
    all_params = []
    luminance_values = []

    for data_set in data_sets:
        initial_png_file, adjusted_png_files, initial_luminance, luminance_file = data_set

        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, initial_luminance, fit_func, luminance_file)

        # Fit sigmoid
        L_values, k_values, x0_values, b_values, r2_avg, r2_scores, delta_Es = adaptator.fit()
        for L, k, x0, b in zip(L_values, k_values, x0_values, b_values):
            all_params.append((L, k, x0, b))
        luminance_values.extend([initial_luminance] * len(L_values))  # Use initial_luminance for fitting

        # Save comparison images using fitted L, k, x0, and b
        output_dir = os.path.join(output_base_dir, f'comparison_low_images_{os.path.basename(os.path.dirname(initial_png_file))}')
        adaptator.save_comparison_images(output_dir, L_values, k_values, x0_values, b_values, r2_scores, delta_Es)

    return all_params, luminance_values

# Paths and data sets
current_path = os.path.abspath(os.path.dirname(__file__))
output_base_dir = os.path.join(current_path, 'data/comparison_images')

# Data sets with initial luminance < 100
data_sets_low_luminance = [
    (os.path.join(current_path, 'data/VW316-TLB/VW316 7CS-RCL.TLB-20220810.HV_25.1441.png'),
     [
         os.path.join(current_path, 'data/VW316-TLB/VW316 7CS-RCL.TLB-20220810.HV_00{}.png'.format(i+1)) for i in range(20)
     ],
     25.1441,
     os.path.join(current_path, 'data/VW316-TLB/sample_luminance.txt')),
    
    (os.path.join(current_path, 'data/VW323-TL/VW323 0CS.TL.HV_49.3145.png'),
     [
         os.path.join(current_path, 'data/VW323-TL/VW323 0CS.TL.HV_00{}.png'.format(i+1)) for i in range(20)
     ],
     49.3145,
     os.path.join(current_path, 'data/VW323-TL/sample_luminance.txt'))
]

# Fit on all data sets separately to get individual L, k, x0, and b values and save comparison images
all_params_low, luminance_values_low = fit_on_all_data_sets(data_sets_low_luminance, "sigmoid", output_base_dir)

# Calculate the average parameters
mean_L = np.mean([param[0] for param in all_params_low])
mean_k = np.mean([param[1] for param in all_params_low])
mean_x0 = np.mean([param[2] for param in all_params_low])
mean_b = np.mean([param[3] for param in all_params_low])

# Paths and data sets for generalized model
generalized_output_base_dir = os.path.join(current_path, 'data/generalized_comparison_images')

def apply_generalized_model(data_sets, L, k, x0, b, output_base_dir):
    all_r2_scores = []
    all_delta_Es = []
    for data_set in data_sets:
        initial_png_file, adjusted_png_files, initial_luminance, luminance_file = data_set
        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, initial_luminance, "sigmoid", luminance_file)
        
        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(adjusted_png_files):
            Y = adaptator.extract_luminance_from_png(y_file)
            X_Ave = adaptator.X_Ave_values[i]

            # Calculate adjusted luminance
            Y_pred = adaptator.apply_brightness_adjustment(adaptator.X, L, k, x0, b)
            
            # Calculate R² and ΔE
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            delta_E = np.sqrt(mse(Y, Y_pred))
            r2_scores.append(r2)
            delta_Es.append(delta_E)

        all_r2_scores.extend(r2_scores)
        all_delta_Es.extend(delta_Es)

        # Save comparison images using generalized L, k, x0, and b
        output_dir = os.path.join(output_base_dir, f'comparison_low_images_{os.path.basename(os.path.dirname(initial_png_file))}')
        adaptator.save_comparison_images(output_dir, L, k, x0, b, r2_scores, delta_Es, generalized=True)

    return all_r2_scores, all_delta_Es

all_r2_scores, all_delta_Es = apply_generalized_model(data_sets_low_luminance, mean_L, mean_k, mean_x0, mean_b, generalized_output_base_dir)

def visualize_params(all_params, initial_luminance, output_file):
    plt.figure(figsize=(10, 6))

    # Compute mean values for each unique luminance
    unique_luminance = np.unique(initial_luminance)
    mean_L = []
    mean_k = []
    mean_x0 = []
    mean_b = []

    for lum in unique_luminance:
        indices = [i for i, x in enumerate(initial_luminance) if x == lum]
        L_values = [all_params[i][0] for i in indices]
        k_values = [all_params[i][1] for i in indices]
        x0_values = [all_params[i][2] for i in indices]
        b_values = [all_params[i][3] for i in indices]

        mean_L.append(np.mean(L_values))
        mean_k.append(np.mean(k_values))
        mean_x0.append(np.mean(x0_values))
        mean_b.append(np.mean(b_values))

    plt.subplot(1, 4, 1)
    plt.scatter(unique_luminance, mean_L)
    plt.xlabel('initial_luminance')
    plt.ylabel('L')
    plt.title('Relationship between L and initial luminance')

    plt.subplot(1, 4, 2)
    plt.scatter(unique_luminance, mean_k)
    plt.xlabel('initial_luminance')
    plt.ylabel('k')
    plt.title('Relationship between k and initial luminance')

    plt.subplot(1, 4, 3)
    plt.scatter(unique_luminance, mean_x0)
    plt.xlabel('initial_luminance')
    plt.ylabel('x0')
    plt.title('Relationship between x0 and initial luminance')

    plt.subplot(1, 4, 4)
    plt.scatter(unique_luminance, mean_b)
    plt.xlabel('initial_luminance')
    plt.ylabel('b')
    plt.title('Relationship between b and initial luminance')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

output_file_low = os.path.join(output_base_dir, 'param_vs_luminance_low.png')
visualize_params(all_params_low, luminance_values_low, output_file_low)

def visualize_r2_deltaE(all_r2_scores, all_delta_Es, output_base_dir):
    plt.figure(figsize=(10, 6))

    # Plot R² scores
    plt.subplot(1, 2, 1)
    plt.hist(all_r2_scores, bins=20, alpha=0.75)
    plt.xlabel('R² Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of R² Scores for Generalized Model')

    # Plot ΔE scores
    plt.subplot(1, 2, 2)
    plt.hist(all_delta_Es, bins=20, alpha=0.75)
    plt.xlabel('ΔE')
    plt.ylabel('Frequency')
    plt.title('Distribution of ΔE Scores for Generalized Model')

    plt.tight_layout()
    output_file = os.path.join(output_base_dir, 'r2_deltaE_distribution.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

visualize_r2_deltaE(all_r2_scores, all_delta_Es, generalized_output_base_dir)

def linear_model(x, m, c):
    return m * x + c

# Fit linear models for the parameters
L_params, _ = curve_fit(linear_model, luminance_values_low, [param[0] for param in all_params_low])
k_params, _ = curve_fit(linear_model, luminance_values_low, [param[1] for param in all_params_low])
x0_params, _ = curve_fit(linear_model, luminance_values_low, [param[2] for param in all_params_low])
b_params, _ = curve_fit(linear_model, luminance_values_low, [param[3] for param in all_params_low])

def apply_linear_model(data_sets, L_params, k_params, x0_params, b_params, output_base_dir):
    all_r2_scores = []
    all_delta_Es = []

    for data_set in data_sets:
        initial_png_file, adjusted_png_files, initial_luminance, luminance_file = data_set
        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, initial_luminance, "sigmoid", luminance_file)
        
        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(adjusted_png_files):
            Y = adaptator.extract_luminance_from_png(y_file)
            X_Ave = adaptator.X_Ave_values[i]

            # Calculate parameters using linear model
            L = linear_model(initial_luminance, *L_params)
            k = linear_model(initial_luminance, *k_params)
            x0 = linear_model(initial_luminance, *x0_params)
            b = linear_model(initial_luminance, *b_params)

            # Calculate adjusted luminance
            Y_pred = adaptator.apply_brightness_adjustment(adaptator.X, L, k, x0, b)
            
            # Calculate R² and ΔE
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            delta_E = np.sqrt(mse(Y, Y_pred))
            r2_scores.append(r2)
            delta_Es.append(delta_E)

        all_r2_scores.extend(r2_scores)
        all_delta_Es.extend(delta_Es)

        # Save comparison images using linear model parameters
        output_dir = os.path.join(output_base_dir, f'comparison_low_images_{os.path.basename(os.path.dirname(initial_png_file))}')
        adaptator.save_comparison_images(output_dir, L, k, x0, b, r2_scores, delta_Es, generalized=True)

    return all_r2_scores, all_delta_Es

all_linear_r2_scores, all_linear_delta_Es = apply_linear_model(data_sets_low_luminance, L_params, k_params, x0_params, b_params, generalized_output_base_dir)

# Visualize the R² and ΔE distributions for the linear model
visualize_r2_deltaE(all_linear_r2_scores, all_linear_delta_Es, generalized_output_base_dir)
