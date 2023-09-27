import subprocess

# Define a function to run a Python file and capture its console output
def run_python_file(file_path):
    try:
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=True)
        print(f"Output of {file_path}:\n{result.stdout}")
        print("-" * 40)  # Print a separator line
    except subprocess.CalledProcessError as e:
        print(f"Error running {file_path}:\n{e.stderr}")

# Run each of your Python files and capture their console output
if __name__ == "__main__":
    # Replace these with the full paths of the Python files you want to run
    file_paths = [
        "C:/Users/chris/Documents/GitHub/Internship-Research/ML models/dt_resampling.py",
        "C:/Users/chris/Documents/GitHub/Internship-Research/ML models/knn_resampling.py",
        "C:/Users/chris/Documents/GitHub/Internship-Research/ML models/rf_resampling.py",
        "C:/Users/chris/Documents/GitHub/Internship-Research/ML models/svc_resampling.py",
        "C:/Users/chris/Documents/GitHub/Internship-Research/ML models/mlp_resampling.py",
    ]

    for file_path in file_paths:
        run_python_file(file_path)


