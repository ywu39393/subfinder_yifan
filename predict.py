import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os


def mc_dropout_predictions_class(model, inputs, label_encoder, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        pred_output = model(inputs, training=True)  # Dropout active during inference
        pred_output = pred_output.numpy().argmax(1)
        pred_classes = label_encoder.inverse_transform(pred_output)  # Decode predicted classes
        predictions.append(pred_classes)
        
    predictions = np.stack(predictions)
    return predictions



# Load the saved model
model_dl = load_model('trained_model.h5')
le = joblib.load('label_encoder.pkl'
                 
                 
############################################
#Import all csv file under Train_data folder
############################################

folder_path = "Predict_data"
# List to hold DataFrames
dataframes = []
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

combined_data = pd.concat(dataframes, ignore_index=True)
combined_data.to_csv('combined_predict_data.csv', index=False)
file_path = os.path.join('combined_predict_data.csv')
data = pd.read_csv(file_path)
                 
                 
                 
#test on first 99 id
test_seqs = np.array([test_item.replace("|", ",").replace(",", " ") for test_item in data["sequence"].values])
test_seqs2 = test_seqs[0:100]
test_seqs2.shape
label = data['cgc_id'][0:100]

result = mc_dropout_predictions_class(model_dl, test_seqs2, le)
t_result = result.T
unique_numbers = np.unique(t_result)  # Get the unique numbers in the array

# Initialize an empty array to store the counts
counts_by_row = np.zeros((t_result.shape[0], len(unique_numbers)), dtype=int)

# Iterate over each row and count the occurrences of each unique number
for i, row in enumerate(t_result):
    counts = np.array([np.sum(row == num) for num in unique_numbers])
    counts_by_row[i] = counts

# Print the counts
print(counts_by_row)
result = pd.DataFrame(counts_by_row, columns = unique_numbers)
result.insert(0, 'ID', label)
result.to_csv('predictions_counts.csv', index=False)
             
             
                 
                 

                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 