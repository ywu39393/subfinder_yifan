# Guide

There are three Python scripts listed below. The first two are used for training a new word embedding model \& best_k_model. The last one is used as a prediction tool for unlabeled data. Detailed descriptions are provided below:

## 01_train_word_embedding_model.py
This script uses supervised and unsupervised sequences to train the word embedding model, which will be used in the next script. If there are new supervised/unsupervised sequences available in the future, you can simply place them into the specific folder under the 'Data_word_embedding' folder and rerun this script. The data format needs to follow the same structure, where the first column is the CGC ID and the second column is the sequence. The column name does not matter. After running this script, the word embedding model will be saved in the 'Embedding_Models' folder. You don't need to create this folder ahead of time; the script will create it for you.

## 02_train_best_k_model.py
This script is used to train the subfinder model. Before training, please place all labeled sequence data into the folder named 'Train_data'. It has to follow the same structure as before, where the first column is the CGC ID and the second column is the sequence. The script will read all the CSV files under this folder and combine them together.

This script also allows you to choose the model and the number of substrates you want to train. Use **python 02_train_best_k_model.py --help** for more detailed instructions. For example, if you want to train a subfinder to identify the top 8 substrate classes using a transformer model, you can type **python 03_train_best_k_model.py -n 7 -m transformer**. The subfinder model and label encoder will be saved as 'trained_model.h5' and 'label_encoder.pkl'.

## 03_predict_with_unlabel_data.py
This script uses the pretrained subfinder model to predict the substrate class based on the output of dbCAN. You need to specify the input file name and the output summary result name. For example, if you have the output from dbCAN named 'cgc_standard.out' and you want the output file named 'predict_summary.csv', you can use **python 03_predict_with_unlabel_data.py -i cgc_standard.out -o predict_summary.csv**, where **-i** represents the input and **-o** represents the output. Use **python 03_predict_with_unlabel_data.py --help** for more detailed instructions. The results will contain three columns: the first column will be the CGC ID, the second column will be the name of the fiber substrate, and the last column will be the score value, which ranges from 0 to 1. The higher the score value, the higher the possibility that this CGC ID belongs to this class.

