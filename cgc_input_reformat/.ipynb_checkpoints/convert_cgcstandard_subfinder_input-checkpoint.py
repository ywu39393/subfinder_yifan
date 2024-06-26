import os
import sys
from collections import defaultdict

def read_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except IOError:
        print(f'Failed to open or read the file: {file_path}')
        sys.exit(-1)
    
    return [line.strip() for line in lines if line.strip()]

def parse_line(line):
    parts = line.split('\t')
    cgc_id = f"{parts[2]}|{parts[0]}"
    gene_type = parts[1]
    annotation = parts[7].replace(' ', '')  # Remove any spaces from annotation
    
    if gene_type == 'null':
        annotation = 'null'
    elif gene_type == 'TC':
        annotation = '.'.join(annotation.split('.')[:3])
    elif gene_type in ['TF', 'STP']:
        annotation = annotation.replace('+', '|')
    # For CAZyme and other cases, keep annotation as is
    
    return cgc_id, annotation

def reformat_cgc(input_file, output_file):
    lines = read_file(input_file)
    cgc_dict = defaultdict(list)
    
    for line in lines[1:]:  # Skip header
        cgc_id, annotation = parse_line(line)
        cgc_dict[cgc_id].append(annotation)
    
    with open(output_file, 'w') as f:
        for cgc_id, annotations in cgc_dict.items():
            f.write(f"{cgc_id}\t{','.join(annotations)}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    reformat_cgc(input_file, output_file)
    
#################################
#Transfer output_file to csv file
#################################
import csv

# Define input and output file paths
input_file_path = output_file  # 
output_file_path = 'output.csv'  

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile)
    
    # Write the header to the CSV file
    writer.writerow(['cgc_id', 'sequence'])
    
    # Process each line from the input file
    for line in reader:
        parts = line.strip().split('\t')  # Split the line into two parts
        writer.writerow(parts)  # Write the parts to the CSV file

print(f"Transformation complete. Output saved to {output_file_path}")

