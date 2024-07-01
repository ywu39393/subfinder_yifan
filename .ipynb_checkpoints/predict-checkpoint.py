import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os



############################################
#number of replication, default=200
n = 200
#input data folder
folder_path = "cgc_input_reformat"
############################################



def mc_dropout_predictions_class(model, inputs, label_encoder, n_samples=200):
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

model_dl = load_model('trained_model.h5')
le = joblib.load('label_encoder.pkl')

############################################
#Import all csv file under Train_data folder
############################################

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
test_seqs = np.array([test_item.replace("|", ",").replace(",", " ") for test_item in data["sequence"].values])
label = data['cgc_id']

############################################
#Run the prediction model
############################################
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
result_melt = pd.melt(result, id_vars='ID', value_vars=column_name, var_name = 'substrate', value_name = 'p-value')
result_melt['p-value'] = result_melt['p-value']/n
result_melt_sort = result_melt.sort_values(by=['ID', 'p-value'], ascending = False)

result_melt_sort.to_csv('predictions_results.csv', index=False)
print(f"Prediction Output saved as predictions_results.csv")