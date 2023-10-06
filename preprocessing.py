import pandas as pd

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('C:/Users/chris/Documents/GitHub/Internship-Research/datasets/Indian Liver Patient.csv')

# Replace 'x' with 1, 'o' with 2, and 'b' with 3 in-place
df = df.dropna()

# The DataFrame df is now modified with the replacements
# You can save it to the original file if needed
df.to_csv('your_file.csv', index=False)
