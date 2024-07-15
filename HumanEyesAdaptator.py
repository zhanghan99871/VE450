import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import cv2 as cv
from skimage.metrics import mean_squared_error as mse
import logging
from RawImage import RawImage
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    def fit(self):
        k_values = []
        b_values = []
        c_values = []
        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(self.Y_files):
            Y = self.extract_luminance_from_png(y_file)
            X_Ave = self.X_Ave_values[i]
            
            # Provide initial guesses and bounds for parameters
            initial_guesses = [-1, 1, 1]
            bounds = ([-20, -20, 0.1], [20, 20, 10])
            
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
            
        r2_avg = np.mean(r2_scores)

        print(f"Fitted parameters: k = {np.mean(k_values)}, b = {np.mean(b_values)}, c = {np.mean(c_values)}")
        print(f"Average R²: {r2_avg}")
        return k_values, b_values, c_values, r2_avg, r2_scores, delta_Es

    def visualize_fit(self, k_values, b_values, c_values, r2_scores, delta_Es, output_file, r2_avg):
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
        
        # Add k, b, c values and average R² to the plot
        plt.figtext(0.5, 0.01, f'Fitted parameters: k = {np.mean(k_values):.2f}, b = {np.mean(b_values):.2f}, c = {np.mean(c_values):.2f} | Average R²: {r2_avg:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        logging.info(f"Figure saved to {output_file}")

    def save_comparison_images(self, output_dir, k_values, b_values, c_values, sample_luminance_values, r2_scores, delta_Es):
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
                adjusted_luminance = self.gamma_function(l, k_values[i], b_values[i], c_values[i], luminance_value)
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
        initial_png_file, adjusted_png_files, initial_luminance = data_set[:3]

        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, fit_func)

        # Fit gamma
        k_values, b_values, c_values, r2_avg, r2_scores, delta_Es = adaptator.fit()
        for k, b, c in zip(k_values, b_values, c_values):
            all_params.append((k, b, c))
        luminance_values.extend(adaptator.X_Ave_values)  # Use X_Ave_values for fitting

        # Save comparison images using fitted k, b, and c
        output_dir = os.path.join(output_base_dir, f'comparison_images_{os.path.basename(os.path.dirname(initial_png_file))}')
        adaptator.save_comparison_images(output_dir, k_values, b_values, c_values, adaptator.X_Ave_values, r2_scores, delta_Es)

        # Visualize R² and ΔE curve for each fit
        adaptator.visualize_fit(k_values, b_values, c_values, r2_scores, delta_Es, os.path.join(output_dir, 'r2_and_delta_e_curve.png'), r2_avg)

    return all_params, luminance_values

def visualize_params(all_params, luminance_values, output_file):
    plt.figure(figsize=(10, 6))

    # Compute mean values for each unique luminance
    unique_luminance = np.unique(luminance_values)
    mean_k = []
    mean_b = []
    mean_c = []

    for lum in unique_luminance:
        indices = [i for i, x in enumerate(luminance_values) if x == lum]
        k_values = [all_params[i][0] for i in indices]
        b_values = [all_params[i][1] for i in indices]
        c_values = [all_params[i][2] for i in indices]

        mean_k.append(np.mean(k_values))
        mean_b.append(np.mean(b_values))
        mean_c.append(np.mean(c_values))

    plt.subplot(1, 3, 1)
    plt.scatter(unique_luminance, mean_k)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('k')
    plt.title('Relationship between k and Luminance')

    plt.subplot(1, 3, 2)
    plt.scatter(unique_luminance, mean_b)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('b')
    plt.title('Relationship between b and Luminance')

    plt.subplot(1, 3, 3)
    plt.scatter(unique_luminance, mean_c)
    plt.xlabel('Luminance (cd/m^2)')
    plt.ylabel('c')
    plt.title('Relationship between c and Luminance')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def fit_relationships(all_params, luminance_values):
    k_values, b_values, c_values = zip(*all_params)
    k_values, b_values, c_values = np.array(k_values).ravel(), np.array(b_values).ravel(), np.array(c_values).ravel()

    # Fit linear models for k, b, and c with respect to luminance
    def linear_model(x, m, c):
        return m * x + c

    k_params, _ = curve_fit(linear_model, luminance_values, k_values)
    b_params, _ = curve_fit(linear_model, luminance_values, b_values)
    c_params, _ = curve_fit(linear_model, luminance_values, c_values)

    return k_params, b_params, c_params

def apply_generalized_model(data_sets, k_params, b_params, c_params, output_base_dir):
    def linear_model(x, m, c):
        return m * x + c

    all_r2_scores = []
    all_delta_Es = []

    for data_set in data_sets:
        initial_png_file, adjusted_png_files, initial_luminance = data_set[:3]
        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, "gamma")
        
        # Calculate k, b, and c using the fitted relationships
        k_value = linear_model(initial_luminance, *k_params)
        b_value = linear_model(initial_luminance, *b_params)
        c_value = linear_model(initial_luminance, *c_params)
        
        # Save adjusted images using best fit parameters
        output_dir = os.path.join(output_base_dir, 'adjusted_with_generalized_model', os.path.basename(initial_png_file).split('.')[0])

        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(adjusted_png_files):
            Y = adaptator.extract_luminance_from_png(y_file)
            X_Ave = adaptator.X_Ave_values[i]

            # Calculate adjusted luminance
            Y_pred = adaptator.gamma_function(adaptator.X, k_value, b_value, c_value, X_Ave)

            # Calculate R² and ΔE
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            delta_E = np.sqrt(mse(Y, Y_pred))
            r2_scores.append(r2)
            delta_Es.append(delta_E)

        all_r2_scores.append(r2_scores)
        all_delta_Es.append(delta_Es)

        adaptator.save_comparison_images(output_dir, [k_value]*len(adjusted_png_files), [b_value]*len(adjusted_png_files), [c_value]*len(adjusted_png_files), adaptator.X_Ave_values, r2_scores, delta_Es)

    return all_r2_scores, all_delta_Es

def visualize_best_fit_results(all_r2_scores, all_delta_Es, output_file, k_params, b_params, c_params, r2_avg):
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
    plt.figtext(0.5, 0.01, f'Fitted parameters relationships: k = {k_params[0]:.2f}*Luminance + {k_params[1]:.2f}, b = {b_params[0]:.2f}*Luminance + {b_params[1]:.2f}, c = {c_params[0]:.2f}*Luminance + {c_params[1]:.2f} | Best Average R²: {r2_avg:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Paths and data sets
current_path = os.path.abspath(os.path.dirname(__file__))
output_base_dir = os.path.join(current_path, 'data/comparison_images')
# Data sets with initial luminance >= 100
data_sets_high_luminance = [
    (os.path.join(current_path, 'data/VW216/VW216.RTSL-BUL.HV_6809.47.png'),
     [
         os.path.join(current_path, 'data/VW216/VW216.RTSL-BUL.HV_00{}.png'.format(i+1)) for i in range(20)
     ],
     6809.47),
    
    (os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_2654.74.png'),
     [
         os.path.join(current_path, 'data/VW310/VW310-6CS.DRL-20220328.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     2654.74),
    
    (os.path.join(current_path, 'data/VW310-PL/VW310-6CS.PL-FTSL-20220401.HV_1744.43.png'),
     [
         os.path.join(current_path, 'data/VW310-PL/VW310-6CS.PL-FTSL-20220401.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     1744.43),
    
    (os.path.join(current_path, 'data/VW316/VW316 7CS.RTSL-BUL-SL-TL.HV_2124.45.png'),
     [
         os.path.join(current_path, 'data/VW316/VW316 7CS.RTSL-BUL-SL-TL.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     2124.45),
    
    (os.path.join(current_path, 'data/VW323/VW323 0CS.SL-RTSL-BUL-RFL.HV_2381.67.png'),
     [
         os.path.join(current_path, 'data/VW323/VW323 0CS.SL-RTSL-BUL-RFL.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     2381.67),
    
    (os.path.join(current_path, 'data/VW326/VW326 0CS.SL-TL-RTSL-BUL-RFL.HV_9001.23.png'),
     [
         os.path.join(current_path, 'data/VW326/VW326 0CS.SL-TL-RTSL-BUL-RFL.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     9001.23),
    
    (os.path.join(current_path, 'data/VW331/VW331_Basic_CHL_simulation setting.DRL_PL_FTSL_20220311.HV_15241.2.png'),
     [
         os.path.join(current_path, 'data/VW331/VW331_Basic_CHL_simulation setting.DRL_PL_FTSL_20220311.HV_00{}.png'.format(i+1)) for i in range(10)
     ],
     15241.2)
]

# Data sets with initial luminance < 100
data_sets_low_luminance = [
    (os.path.join(current_path, 'data/VW316-TLB/VW316 7CS-RCL.TLB-20220810.HV_25.1441.png'),
     [
         os.path.join(current_path, 'data/VW316-TLB/VW316 7CS-RCL.TLB-20220810.HV_00{}.png'.format(i+1)) for i in range(20)
     ],
     25.1441),
    
    (os.path.join(current_path, 'data/VW323-TL/VW323 0CS.TL.HV_49.3145.png'),
     [
         os.path.join(current_path, 'data/VW323-TL/VW323 0CS.TL.HV_00{}.png'.format(i+1)) for i in range(20)
     ],
     49.3145)
]

def visualize_predictions(data_sets, mean_k, mean_b, mean_c, output_base_dir, fit_type='average', group_name=''):
    all_mean_r2_scores = []
    all_mean_delta_Es = []

    for data_set in data_sets:
        initial_png_file, adjusted_png_files, initial_luminance = data_set[:3]
        adaptator = HumanEyesAdaptator(initial_png_file, adjusted_png_files, "gamma")
        
        path_parts = initial_png_file.split(os.sep)
        data_index = path_parts.index('data')
        first_dir = path_parts[data_index + 1]
        second_dir = path_parts[data_index + 2]

        output_dir = os.path.join(output_base_dir, f'predicted_vs_actual_{group_name}', first_dir, second_dir)

        r2_scores = []
        delta_Es = []

        for i, y_file in enumerate(adjusted_png_files):
            Y = adaptator.extract_luminance_from_png(y_file)
            X_Ave = adaptator.X_Ave_values[i]

            if fit_type == 'linear':
                k_value = mean_k[0] * initial_luminance + mean_k[1]
                b_value = mean_b[0] * initial_luminance + mean_b[1]
                c_value = mean_c[0] * initial_luminance + mean_c[1]
            else:
                k_value = mean_k
                b_value = mean_b
                c_value = mean_c

            # Calculate adjusted luminance using mean parameters
            Y_pred = adaptator.gamma_function(adaptator.X, k_value, b_value, c_value, X_Ave)

            # Calculate R² and ΔE
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            delta_E = np.sqrt(mse(Y, Y_pred))
            r2_scores.append(r2)
            delta_Es.append(delta_E)

        all_mean_r2_scores.append(np.mean(r2_scores))
        all_mean_delta_Es.append(np.mean(delta_Es))

        adaptator.save_comparison_images(output_dir, [k_value]*len(adjusted_png_files), [b_value]*len(adjusted_png_files), [c_value]*len(adjusted_png_files), adaptator.X_Ave_values, r2_scores, delta_Es)

    visualize_model_performance(all_mean_r2_scores, all_mean_delta_Es, output_base_dir, group_name)

def visualize_model_performance(all_mean_r2_scores, all_mean_delta_Es, output_base_dir, group_name):
    # Plot mean R² values
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(all_mean_r2_scores) + 1), all_mean_r2_scores, marker='o')
    for i, txt in enumerate(all_mean_r2_scores):
        plt.annotate(f"{txt:.4f}", (i+1, all_mean_r2_scores[i]))
    plt.xlabel('Dataset Index')
    plt.ylabel('Mean R² Score')
    plt.title(f'Mean R² Scores for Each Adjusted Dataset ({group_name})')
    
    # Annotate with k, b, c parameters and average R²
    if group_name == 'high_luminance':
        mean_k, mean_b, mean_c = mean_k_high, mean_b_high, mean_c_high
        avg_r2 = np.mean(all_mean_r2_scores)
        avg_deltaE = np.mean(all_mean_delta_Es)
        plt.figtext(0.5, 0.01, f'Mean k: {mean_k:.2f}, Mean b: {mean_b:.2f}, Mean c: {mean_c:.2f} | Avg R²: {avg_r2:.2f} | Avg ΔE: {avg_deltaE:.2f}', ha='center', fontsize=10)
    elif group_name == 'low_luminance':
        k_slope, k_intercept = k_params_low
        b_slope, b_intercept = b_params_low
        c_slope, c_intercept = c_params_low
        avg_r2 = np.mean(all_mean_r2_scores)
        avg_deltaE = np.mean(all_mean_delta_Es)
        plt.figtext(0.5, 0.01, f'k: {k_slope:.2f}*Luminance + {k_intercept:.2f}, b: {b_slope:.2f}*Luminance + {b_intercept:.2f}, c: {c_slope:.2f}*Luminance + {c_intercept:.2f} | Avg R²: {avg_r2:.2f} | Avg ΔE: {avg_deltaE:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_file_r2 = os.path.join(output_base_dir, f'mean_r2_performance_{group_name}.png')
    plt.savefig(output_file_r2, dpi=300)
    plt.close()
    logging.info(f"Mean R² performance figure saved to {output_file_r2}")

    # Plot mean ΔE values
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(all_mean_delta_Es) + 1), all_mean_delta_Es, marker='o')
    for i, txt in enumerate(all_mean_delta_Es):
        plt.annotate(f"{txt:.4f}", (i+1, all_mean_delta_Es[i]))
    plt.xlabel('Dataset Index')
    plt.ylabel('Mean ΔE')
    plt.title(f'Mean ΔE for Each Adjusted Dataset ({group_name})')
    
    # Annotate with k, b, c parameters and average ΔE
    if group_name == 'high_luminance':
        mean_k, mean_b, mean_c = mean_k_high, mean_b_high, mean_c_high
        avg_r2 = np.mean(all_mean_r2_scores)
        avg_deltaE = np.mean(all_mean_delta_Es)
        plt.figtext(0.5, 0.01, f'Mean k: {mean_k:.2f}, Mean b: {mean_b:.2f}, Mean c: {mean_c:.2f} | Avg R²: {avg_r2:.2f} | Avg ΔE: {avg_deltaE:.2f}', ha='center', fontsize=10)
    elif group_name == 'low_luminance':
        k_slope, k_intercept = k_params_low
        b_slope, b_intercept = b_params_low
        c_slope, c_intercept = c_params_low
        avg_r2 = np.mean(all_mean_r2_scores)
        avg_deltaE = np.mean(all_mean_delta_Es)
        plt.figtext(0.5, 0.01, f'k: {k_slope:.2f}*Luminance + {k_intercept:.2f}, b: {b_slope:.2f}*Luminance + {b_intercept:.2f}, c: {c_slope:.2f}*Luminance + {c_intercept:.2f} | Avg R²: {avg_r2:.2f} | Avg ΔE: {avg_deltaE:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_file_deltaE = os.path.join(output_base_dir, f'mean_deltaE_performance_{group_name}.png')
    plt.savefig(output_file_deltaE, dpi=300)
    plt.close()
    logging.info(f"Mean ΔE performance figure saved to {output_file_deltaE}")

# Fit on all data sets separately to get individual k, b, and c values and save comparison images
all_params_high, luminance_values_high = fit_on_all_data_sets(data_sets_high_luminance, "gamma", output_base_dir)
all_params_low, luminance_values_low = fit_on_all_data_sets(data_sets_low_luminance, "gamma", output_base_dir)

# Visualize and save the relationship between parameters and luminance for both high and low luminance data sets
output_file_high = os.path.join(output_base_dir, 'param_vs_luminance_high.png')
visualize_params(all_params_high, luminance_values_high, output_file_high)

output_file_low = os.path.join(output_base_dir, 'param_vs_luminance_low.png')
visualize_params(all_params_low, luminance_values_low, output_file_low)

# Calculate the average parameters for high luminance data sets
# mean_k_high = np.mean([param[0] for param in all_params_high])
# mean_b_high = np.mean([param[1] for param in all_params_high])
# mean_c_high = np.mean([param[2] for param in all_params_high])

# Fit linear models for low luminance data sets
# k_params_low, b_params_low, c_params_low = fit_relationships(all_params_low, luminance_values_low)

# Apply the generalized model and visualize predictions
# visualize_predictions(data_sets_high_luminance, mean_k_high, mean_b_high, mean_c_high, output_base_dir, fit_type='average', group_name='high_luminance')
# visualize_predictions(data_sets_low_luminance, k_params_low, b_params_low, c_params_low, output_base_dir, fit_type='linear', group_name='low_luminance')
