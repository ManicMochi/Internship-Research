import pandas as pd

# Load your CSV file into a DataFrame
# Replace 'your_dataset.csv' with the actual path to your CSV file
df = pd.read_csv('C:/Users/chris/Documents/GitHub/Internship-Research/datasets/Ecoli.csv')

# Create a mapping dictionary for class labels to numbers
class_mapping = {'cp': 1, 'im': 2, 'imU': 3, 'om': 4, 'pp': 5}

# Replace class labels with numbers in the 'class' column
df['class'].replace(class_mapping, inplace=True)

# Save the updated dataframe to a new CSV file
df.to_csv('updated_dataset.csv', index=False)

print("Updated dataset saved to 'updated_dataset.csv'")