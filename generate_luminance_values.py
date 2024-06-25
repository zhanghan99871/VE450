import numpy as np

class LuminanceGenerator:
    def __init__(self, initial_luminance):
        self.initial_luminance = initial_luminance

    def generate_sample_luminance_values(self, num_samples=30):
        # Define a range that is more centered around the initial luminance value
        min_luminance = self.initial_luminance * 0.01  # TODO: Adjust this factor as needed
        max_luminance = self.initial_luminance * 100   # TODO: Adjust this factor as needed
        # Use np.logspace to generate values evenly distributed on a log scale
        sample_luminance_values = np.logspace(np.log10(min_luminance), np.log10(max_luminance), num_samples)
        return sample_luminance_values

    def print_sample_luminance_values(self):
        sample_luminance_values = self.generate_sample_luminance_values()
        print("Sample Luminance Values:")
        for value in sample_luminance_values:
            print(f"{value:.6g} cd/m²")

# Example usage
initial_luminance = 6809.47  # TODO: Initial luminance in cd/m²
luminance_generator = LuminanceGenerator(initial_luminance)
luminance_generator.print_sample_luminance_values()
