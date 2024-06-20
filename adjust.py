import pandas as pd
import numpy as np

def apply_brightness_adjustment(df, X_Max, k, b, X_Ave, epsilon=1e-10):
    # Apply the brightness adjustment formula
    df = df + epsilon  # Add epsilon to avoid log10(0)
    Y = 100 * np.log10(1 + 9 * (df / X_Max) ** (k * np.log10(df * X_Ave) + b))
    
    return Y

# Replace with the actual file path
file_path = '/home/yaqing/ve450/Human_eye-Adaptation-Rendering-Algorithm/output_file.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Parameters for the formula
X_Max = df.max().max()  # Assuming the maximum value in the DataFrame is X_Max
k = 0.0155  # Example value from your formula fitting
b = -0.1085  # Example value from your formula fitting
X_Ave = 100  # Set this to the specific luminance value you have

# Apply the brightness adjustment
adjusted_df = apply_brightness_adjustment(df, X_Max, k, b, X_Ave)

# Save the adjusted data to a new CSV file
output_file_path = 'adjusted_output_file.csv'
adjusted_df.to_csv(output_file_path, index=False)

print(f"Brightness adjusted data has been successfully written to {output_file_path}")
