import sys
import os
sys.path.append('/content/drive/MyDrive/data')

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
import logging
from RawImage import RawImage
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RawImage:
    def loadRGB(self, png_file):
        if not os.path.exists(png_file):
            raise FileNotFoundError(f"File {png_file} not found.")

        img = cv.imread(png_file)
        if img is None:
            raise ValueError(f"Failed to read the image file {png_file}.")

        self.rgb = np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    def convert_rgb_to_lab_luminance(self):
        lab = cv.cvtColor(self.rgb, cv.COLOR_RGB2LAB)
        self.luminance = lab[:, :, 0]

class HumanEyesAdaptator:
    def __init__(self, initial_png_file, adjusted_png_files, fit_func):
        self.X = self.extract_luminance_from_png(initial_png_file)
        self.Y_files = adjusted_png_files
        self.X_Max = self.X.max()
        self.fit_func = fit_func
        self.X_Ave_values = self.calculate_X_Ave_values()

    def extract_luminance_from_png(self, png_file):
        image = RawImage()
        image.loadRGB(png_file)
        image.convert_rgb_to_lab_luminance()
        return image.luminance

    def calculate_X_Ave_values(self):
        X_Ave_values = []
        for y_file in self.Y_files:
            image = RawImage()
            image.loadRGB(y_file)
            image.convert_rgb_to_lab_luminance()
            X_Ave_values.append(np.mean(image.luminance))
        logging.info(f"Calculated X_Ave values for adjusted images: {X_Ave_values}")
        return X_Ave_values

    def sigmoid_function(self, X, a, b, c, d, X_Ave):
        X = np.clip(X, -500, 500)
        return a / (1 + np.exp(-(X - (b + X_Ave)) / c)) + d

    def fit(self):
        a_values = []
        b_values = []
        c_values = []
        d_values = []
        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(self.Y_files):
            Y = self.extract_luminance_from_png(y_file)
            X_Ave = self.X_Ave_values[i]

            # Provide initial guesses and bounds for parameters
            initial_guesses = [150, 50, 20, 1]
            bounds = ([100, 50, 10, 0], [255, 255, 255, 10])

            try:
                params, _ = curve_fit(
                    lambda X, a, b, c, d: self.sigmoid_function(X, a, b, c, d, X_Ave),
                    self.X.ravel(), Y.ravel(),
                    p0=initial_guesses, bounds=bounds
                )
                a_values.append(params[0])
                b_values.append(params[1])
                c_values.append(params[2])
                d_values.append(params[3])

                # Calculate R²
                Y_pred = self.sigmoid_function(self.X, params[0], params[1], params[2], params[3], X_Ave)
                r2 = r2_score(Y.ravel(), Y_pred.ravel())
                r2_scores.append(r2)

                # Calculate ΔE
                delta_E = np.sqrt(mse(Y, Y_pred))
                delta_Es.append(delta_E)
            except RuntimeError as e:
                print(f"Fit did not converge for file {y_file}: {e}")
                continue

        r2_avg = np.mean(r2_scores)

        print(f"Fitted parameters: a = {np.mean(a_values)}, b = {np.mean(b_values)}, c = {np.mean(c_values)}, d = {np.mean(d_values)}")
        print(f"Average R²: {r2_avg}")
        return a_values, b_values, c_values, d_values, r2_avg, r2_scores, delta_Es

    def visualize_fit(self, a_values, b_values, c_values, d_values, r2_scores, delta_Es, output_file, r2_avg):
        plt.figure(figsize=(10, 6))

        # Plot R² values
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o')
        for i, txt in enumerate(r2_scores):
            plt.annotate(f"{txt:.4f}", (i+1, r2_scores[i]))
        plt.xlabel('Image Index')
        plt.ylabel('R² Score')
        plt.title('R² Scores for Each Adjusted Image')

        # Plot ΔE values
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(delta_Es) + 1), delta_Es, marker='o')
        for i, txt in enumerate(delta_Es):
            plt.annotate(f"{txt:.4f}", (i+1, delta_Es[i]))
        plt.xlabel('Image Index')
        plt.ylabel('ΔE')
        plt.title('ΔE for Each Adjusted Image')

        # Add a, b, c, d values and average R² to the plot
        plt.figtext(0.5, 0.01, f'Fitted parameters: a = {np.mean(a_values):.2f}, b = {np.mean(b_values):.2f}, c = {np.mean(c_values):.2f}, d = {np.mean(d_values):.2f} | Average R²: {r2_avg:.2f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        logging.info(f"Figure saved to {output_file}")

    def save_comparison_images(self, output_dir, a_values, b_values, c_values, d_values, sample_luminance_values, r2_scores, delta_Es):
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
                l, a_channel, b_channel = cv.split(lab_original)

                # Ensure that the original L channel is in the correct range
                logging.info(f"Original L channel min: {l.min()}, max: {l.max()}")
                # Apply brightness adjustment to the L channel
                adjusted_luminance = self.sigmoid_function(l, a_values[i], b_values[i], c_values[i], d_values[i], luminance_value)
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
                lab_adjusted = cv.merge([adjusted_luminance, a_channel, b_channel])
                # Convert back to BGR color space
                adjusted_img = cv.cvtColor(lab_adjusted, cv.COLOR_LAB2BGR)

                # Combine images side by side
                comparison_img = np.hstack((original_img, adjusted_img))

                # Add title with the luminance value, R² score, and ΔE
                title = f"Sample Luminance: X_AVE {luminance_value:.2f} , R^2: {r2_score_val:.2f}, DeltaE: {delta_E_val:.2f}"
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
        initial_png_file, adjusted_png_files = data_set

        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, fit_func)

        # Fit sigmoid
        a_values, b_values, c_values, d_values, r2_avg, r2_scores, delta_Es = adaptator.fit()
        for a, b, c, d in zip(a_values, b_values, c_values, d_values):
            all_params.append((a, b, c, d))
        luminance_values.extend(adaptator.X_Ave_values)  # Use X_Ave_values for fitting

        # Save comparison images using fitted a, b, c, d
        output_dir = os.path.join(output_base_dir, f'comparison_images_{os.path.basename(os.path.dirname(initial_png_file))}')
        adaptator.save_comparison_images(output_dir, a_values, b_values, c_values, d_values, adaptator.X_Ave_values, r2_scores, delta_Es)

        # Visualize R² and ΔE curve for each fit
        adaptator.visualize_fit(a_values, b_values, c_values, d_values, r2_scores, delta_Es, os.path.join(output_dir, 'r2_and_delta_e_curve.png'), r2_avg)

    return all_params, luminance_values

def visualize_params(all_params, luminance_values, output_file):
    plt.figure(figsize=(10, 6))

    # Compute mean values for each unique luminance
    unique_luminance = np.unique(luminance_values)
    mean_a = []
    mean_b = []
    mean_c = []
    mean_d = []

    for lum in unique_luminance:
        indices = [i for i, x in enumerate(luminance_values) if x == lum]
        a_values = [all_params[i][0] for i in indices]
        b_values = [all_params[i][1] for i in indices]
        c_values = [all_params[i][2] for i in indices]
        d_values = [all_params[i][3] for i in indices]

        mean_a.append(np.mean(a_values))
        mean_b.append(np.mean(b_values))
        mean_c.append(np.mean(c_values))
        mean_d.append(np.mean(d_values))

    plt.subplot(1, 4, 1)
    plt.scatter(unique_luminance, mean_a)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('a')
    plt.title('Relationship between a and Luminance')

    plt.subplot(1, 4, 2)
    plt.scatter(unique_luminance, mean_b)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('b')
    plt.title('Relationship between b and Luminance')

    plt.subplot(1, 4, 3)
    plt.scatter(unique_luminance, mean_c)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('c')
    plt.title('Relationship between c and Luminance')

    plt.subplot(1, 4, 4)
    plt.scatter(unique_luminance, mean_d)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('d')
    plt.title('Relationship between d and Luminance')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def fit_relationships(all_params, luminance_values):
    a_values, b_values, c_values, d_values = zip(*all_params)
    a_values, b_values, c_values, d_values = np.array(a_values).ravel(), np.array(b_values).ravel(), np.array(c_values).ravel(), np.array(d_values).ravel()

    # Fit linear models for a, b, c, and d with respect to luminance
    def linear_model(x, m, c):
        return m * x + c

    a_params, _ = curve_fit(linear_model, luminance_values, a_values)
    b_params, _ = curve_fit(linear_model, luminance_values, b_values)
    c_params, _ = curve_fit(linear_model, luminance_values, c_values)
    d_params, _ = curve_fit(linear_model, luminance_values, d_values)

    return a_params, b_params, c_params, d_params

def apply_generalized_model(data_sets, a_params, b_params, c_params, d_params, output_base_dir):
    def linear_model(x, m, c):
        return m * x + c

    all_r2_scores = []
    all_delta_Es = []

    for data_set in data_sets:
        initial_png_file, adjusted_png_files = data_set
        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, "sigmoid")

        # Calculate a, b, c, and d using the fitted relationships
        a_value = linear_model(adaptator.X_Ave_values[0], *a_params)
        b_value = linear_model(adaptator.X_Ave_values[0], *b_params)
        c_value = linear_model(adaptator.X_Ave_values[0], *c_params)
        d_value = linear_model(adaptator.X_Ave_values[0], *d_params)

        # Save adjusted images using best fit parameters
        output_dir = os.path.join(output_base_dir, 'adjusted_with_generalized_model', os.path.basename(initial_png_file).split('.')[0])

        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(adjusted_png_files):
            Y = adaptator.extract_luminance_from_png(y_file)
            X_Ave = adaptator.X_Ave_values[i]

            # Calculate adjusted luminance
            Y_pred = adaptator.sigmoid_function(adaptator.X, a_value, b_value, c_value, d_value, X_Ave)

            # Calculate R² and ΔE
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            delta_E = np.sqrt(mse(Y, Y_pred))
            r2_scores.append(r2)
            delta_Es.append(delta_E)

        all_r2_scores.append(r2_scores)
        all_delta_Es.append(delta_Es)

        adaptator.save_comparison_images(output_dir, [a_value]*len(adjusted_png_files), [b_value]*len(adjusted_png_files), [c_value]*len(adjusted_png_files), [d_value]*len(adjusted_png_files), adaptator.X_Ave_values, r2_scores, delta_Es)

    return all_r2_scores, all_delta_Es

def visualize_best_fit_results(all_r2_scores, all_delta_Es, output_file, a_params, b_params, c_params, d_params, r2_avg):
    plt.figure(figsize=(10, 6))

    for r2_scores, delta_Es in zip(all_r2_scores, all_delta_Es):
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

    # Add best parameters and average R² to the plot
    plt.figtext(0.5, 0.01, f'Fitted parameters relationships: a = {a_params[0]:.2f}*Luminance + {a_params[1]:.2f}, b = {b_params[0]:.2f}*Luminance + {b_params[1]:.2f}, c = {c_params[0]:.2f}*Luminance + {c_params[1]:.2f}, d = {d_params[0]:.2f}*Luminance + {d_params[1]:.2f} | Best Average R²: {r2_avg:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Paths and data sets
current_path = '/home/yaqing/ve450/Human_eye-Adaptation-Rendering-Algorithm/data'
output_base_dir = os.path.join(current_path, 'comparison_images')
data_sets = [
    (os.path.join(current_path, 'VW216/VW216.RTSL-BUL.HV_6809.47.png'),
     [os.path.join(current_path, 'VW216/VW216.RTSL-BUL.HV_00{}.png'.format(i+1)) for i in range(20)]),

    (os.path.join(current_path, 'VW310/VW310-6CS.DRL-20220328.HV_2654.74.png'),
     [os.path.join(current_path, 'VW310/VW310-6CS.DRL-20220328.HV_00{}.png'.format(i+1)) for i in range(10)
     ]),

    (os.path.join(current_path, 'VW310-PL/VW310-6CS.PL-FTSL-20220401.HV_1744.43.png'),
     [os.path.join(current_path, 'VW310-PL/VW310-6CS.PL-FTSL-20220401.HV_00{}.png'.format(i+1)) for i in range(10)
     ]),

    (os.path.join(current_path, 'VW316/VW316 7CS.RTSL-BUL-SL-TL.HV_2124.45.png'),
     [os.path.join(current_path, 'VW316/VW316 7CS.RTSL-BUL-SL-TL.HV_00{}.png'.format(i+1)) for i in range(10)
     ]),

    (os.path.join(current_path, 'VW323/VW323 0CS.SL-RTSL-BUL-RFL.HV_2381.67.png'),
     [
         os.path.join(current_path, 'VW323/VW323 0CS.SL-RTSL-BUL-RFL.HV_00{}.png'.format(i+1)) for i in range(10)
     ]),

    (os.path.join(current_path, 'VW326/VW326 0CS.SL-TL-RTSL-BUL-RFL.HV_9001.23.png'),
     [
         os.path.join(current_path, 'VW326/VW326 0CS.SL-TL-RTSL-BUL-RFL.HV_00{}.png'.format(i+1)) for i in range(10)
     ]),

    (os.path.join(current_path, 'VW331/VW331_Basic_CHL_simulation setting.DRL_PL_FTSL_20220311.HV_15241.2.png'),
     [
         os.path.join(current_path, 'VW331/VW331_Basic_CHL_simulation setting.DRL_PL_FTSL_20220311.HV_00{}.png'.format(i+1)) for i in range(10)
     ])
]

# Fit on all data sets
all_params, luminance_values = fit_on_all_data_sets(data_sets, "sigmoid", output_base_dir)

output_file = os.path.join(output_base_dir, 'param_vs_luminance.png')
visualize_params(all_params, luminance_values, output_file)
