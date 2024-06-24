import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from RawImage import RawImage

class HumanEyesAdaptator:
    def __init__(self, initial_txt_file, adjusted_txt_files):
        self.X = self.extract_luminance(initial_txt_file)
        self.Y_files = adjusted_txt_files
        self.X_Max = self.X.max()

    def extract_luminance(self, txt_file):
        image = RawImage(txt_file)
        image.loadLuminance(txt_file)
        return image.luminance[:, :, 0]

    def apply_brightness_adjustment(self, df, X_Ave, k, b, epsilon=1e-10):
        df = df + epsilon  # Add epsilon to avoid log10(0)
        Y = 100 * np.log10(1 + 9 * (df / self.X_Max) ** (k * np.log10(df * X_Ave) + b))
        return Y

    def gamma_function(self, X, k, b, X_Ave):
        epsilon = 1e-10
        X = X + epsilon  # Add epsilon to avoid log10(0)
        return 100 * np.log10(1 + 9 * (X / self.X_Max) ** (k * np.log10(X * X_Ave) + b))

    def fit_gamma(self):
        k_values = []
        b_values = []

        for i, y_file in enumerate(self.Y_files):
            Y = self.extract_luminance(y_file)
            X_Ave = self.get_X_Ave_for_file(y_file)
            
            params, _ = curve_fit(lambda X, k, b: self.gamma_function(X, k, b, X_Ave), self.X.ravel(), Y.ravel())
            k_values.append(params[0])
            b_values.append(params[1])
        
        k_avg = np.mean(k_values)
        b_avg = np.mean(b_values)

        print(f"Fitted parameters: k = {k_avg}, b = {b_avg}")
        return k_avg, b_avg

    def get_X_Ave_for_file(self, y_file):
        # Replace with the actual method to extract X_Ave from the file or set it manually
        # Assuming X_Ave is provided in the same file in a specific line format
        with open(y_file, 'r') as file:
            lines = file.readlines()
            # Assuming X_Ave is in a specific line, e.g., the 7th line
            X_Ave = float(lines[6].strip())
        return X_Ave

# Example usage
initial_txt_file = 'path_to_initial_file.txt'
adjusted_txt_files = ['path_to_adjusted_file1.txt', 'path_to_adjusted_file2.txt', ...]  # Add all paths to adjusted files

adaptator = HumanEyesAdaptator(initial_txt_file, adjusted_txt_files)
k, b = adaptator.fit_gamma()

# Apply the brightness adjustment using the fitted parameters
for y_file in adjusted_txt_files:
    X_Ave = adaptator.get_X_Ave_for_file(y_file)
    Y = adaptator.extract_luminance(y_file)
    adjusted_Y = adaptator.apply_brightness_adjustment(Y, X_Ave, k, b)
    
    output_file_path = y_file.replace('.txt', '_adjusted.txt')
    np.savetxt(output_file_path, adjusted_Y, fmt='%.10f')

    print(f"Brightness adjusted data has been successfully written to {output_file_path}")
