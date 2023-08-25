import pandas as pd

# Load your CSV file into a DataFrame
# Replace 'your_dataset.csv' with the actual path to your CSV file
df = pd.read_csv('C:/Users/chris/Downloads/VSCode/Internship_objects/datasets/creditcard.csv')

# Assuming 'y' column contains the class labels
class_counts = df['Class'].value_counts()
print(class_counts)
