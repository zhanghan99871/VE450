import pandas as pd

def read_xmp_export(file_path):
    # Initialize lists to store data
    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip initial non-data lines (up to row 9) and read data from row 10 onwards
    for line in lines[9:]:
        parts = line.strip().split()
        if parts and all(part.replace('.', '', 1).isdigit() for part in parts):  # Check if all parts are numeric
            data.append([float(val) for val in parts])

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    return df

# Replace with actual file path
file_path = '/home/yaqing/ve450/VW216.RTSL-BUL.HV_001.txt'
df = read_xmp_export(file_path)
print(df.head())

# Save to Excel
excel_path = 'output_file.xlsx'
df.to_excel(excel_path, index=False)

print(f"Data has been successfully written to {excel_path}")
