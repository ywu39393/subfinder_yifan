import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import sys
from collections import defaultdict
from Codes.import_data import reformat_cgc
import csv


############################################
#initialization
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Process some inputs.")

# Add the arguments
parser.add_argument("-i", "--input", type=str, default="cgc_standard.out", help="Input data file name (default: 'cgc_standard.out')")
parser.add_argument("-o", "--output", type=str, default="predict_summary.csv", help="Model prediction results summary (default: 'predict_summary.csv')")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_file = args.input
output_file_location = args.output

# Your code using input_file and output_file
# Example: Read input_file and write results to output_file

print(f"Input data file: {input_file}")
print(f"Output summary file: {output_file_location}")
############################################

###########tranform cgc input to required input on predict model

output_file = 'reformat'
reformat_cgc(input_file, output_file)
    

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
#remove files        
os.remove('reformat')



#############predict function############
def mc_dropout_predictions_class(model, inputs, label_encoder, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        pred_output = model(inputs, training=True)  # Dropout active during inference
        pred_output = pred_output.numpy().argmax(1)
        pred_classes = label_encoder.inverse_transform(pred_output)  # Decode predicted classes
        predictions.append(pred_classes)
        
    predictions = np.stack(predictions)
    return predictions
############################################
# Load the saved model
############################################

model_dl = load_model('trained_model_best.h5')
le = joblib.load('label_encoder_best.pkl')

############################################
#Import csv file 
############################################

data = pd.read_csv(output_file_path)
test_seqs = np.array([test_item.replace("|", ",").replace(",", " ") for test_item in data["sequence"].values])
label = data['cgc_id']
os.remove(output_file_path)

############################################
#Run the prediction model
############################################
n = 100
result = mc_dropout_predictions_class(model_dl, test_seqs, le, n)
t_result = result.T
unique_numbers = np.unique(t_result)  # Get the unique numbers in the array

# Initialize an empty array to store the counts
counts_by_row = np.zeros((t_result.shape[0], len(unique_numbers)), dtype=int)

# Iterate over each row and count the occurrences of each unique number
for i, row in enumerate(t_result):
    counts = np.array([np.sum(row == num) for num in unique_numbers])
    counts_by_row[i] = counts


result = pd.DataFrame(counts_by_row, columns = unique_numbers)
result.insert(0, 'ID', label)


column_name = result.columns.values[1:]
result_melt = pd.melt(result, id_vars='ID', value_vars=column_name, var_name = 'substrate', value_name = 'score')
result_melt['score'] = result_melt['score']/n
result_melt_sort = result_melt.sort_values(by=['ID', 'score'], ascending = False)
#reorganizing
result_melt_sort = result_melt_sort[~(result_melt_sort['score'] == 0)]

result_melt_sort.to_csv(output_file_location, index=False)
print(f"Prediction Output saved as {output_file_location}")