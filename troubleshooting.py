import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('C:/Users/chris/Documents/GitHub/Internship-Research/datasets/creditcard.csv')

# Assuming your class column is named 'class', adjust as needed
class_counts = df['Class'].value_counts()

print("Class Counts:")
for class_label, count in class_counts.items():
    print(f"{class_label}: {count}")
