import re

# Define the input and output file names
input_file = 'page_blocks.csv'
output_file = 'page_blocks_cleaned.csv'

# Open the input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Iterate through each line in the input file
    for line in infile:
        # Remove quotation marks while preserving commas
        cleaned_line = line.replace('"', '')
        outfile.write(cleaned_line)

print(f'Cleaning from {input_file} to {output_file} completed successfully.')