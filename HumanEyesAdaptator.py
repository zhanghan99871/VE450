import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from RawImage import RawImage
from generate_luminance_values import LuminanceGenerator

class HumanEyesAdaptator:
    def __init__(self, initial_txt_file, adjusted_txt_files, initial_luminance):
        self.X = self.extract_luminance(initial_txt_file)
        self.Y_files = adjusted_txt_files
        self.X_Max = self.X.max()
        self.initial_luminance = initial_luminance 
        self.luminance_generator = LuminanceGenerator(self.initial_luminance)
        self.X_Ave_values = self.generate_sample_luminance_values()

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
        r2_scores = []

        for i, y_file in enumerate(self.Y_files):
            Y = self.extract_luminance(y_file)
            X_Ave = self.X_Ave_values[i]
            
            params, _ = curve_fit(lambda X, k, b: self.gamma_function(X, k, b, X_Ave), self.X.ravel(), Y.ravel())
            k_values.append(params[0])
            b_values.append(params[1])
            
            # Calculate R²
            Y_pred = self.gamma_function(self.X, params[0], params[1], X_Ave)
            r2 = r2_score(Y.ravel(), Y_pred.ravel())
            r2_scores.append(r2)
        
        k_avg = np.mean(k_values)
        b_avg = np.mean(b_values)
        r2_avg = np.mean(r2_scores)

        print(f"Fitted parameters: k = {k_avg}, b = {b_avg}")
        print(f"Average R²: {r2_avg}")
        return k_avg, b_avg, r2_avg

    def generate_sample_luminance_values(self):
        return self.luminance_generator.generate_sample_luminance_values()

# TODO: replace with actual file paths
initial_txt_file = '/path/to/initial.txt'
adjusted_txt_files = [
    '/path/to/adjust1.txt','/path/to/adjust2.txt', #...
]
initial_luminance = 6809.47  # TODO: Initial luminance in cd/m²

adaptator = HumanEyesAdaptator(initial_txt_file, adjusted_txt_files, initial_luminance)

# Fit gamma
k, b, r2_avg = adaptator.fit_gamma()
    