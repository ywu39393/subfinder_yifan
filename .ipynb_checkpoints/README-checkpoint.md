# 
There are three python script were list below, the first two were used for training new word embedding model/best_k_model. The last one was used as prediction tool for unlabeled data. Detail description list below:
## 01_train_word_embedding_model.py
This script used the supervised sequences and unsupervised sequence to train the word embedding model which will be used on the next step. If there is new supervised/unsupervised sequence available in the future, you can just put them into the specific folder under the 'Data_word_embedding' folder and rerun this script. The data format need follow the same structure where the first column is cgc id and the second column was sequence. After running this script, the word embedding model will be saved under folder 'Embedding_Models'. There is no need to create this folder ahead, the script will create this folder for you.

## 02_train_best_k_model.py
This script can be used to training the subfinder model, first before training, please put all labeled sequence data into the folder named 'Train_data', it has to follow the same structure as before where the first column is cgc id and the second column was sequence. The script will read all the csv file under this folder and combine them together. 
This script also give you the ability to choose the model and number of substrate you want train. Using **python 02_train_best_k_model.py --help** for more detailed description. For example, if I want to traing a subfinder to indentify the top 8 substrate class using transformer model, I can type **python 03_train_best_k_model.py -n 7 -m transformer**. The subfinder model and label encoder will be saved as 'trained_model.h5' and 'label_encoder.pkl'.


## 03_predict_with_unlabel_data.py

This script was used the pretrained subfinder model to predict the substrate class based on the output of dbCAN. You need to specific the input file name and output summary reuslt name. For example if the you had the output got from dbCAN named 'cgc_standard.out' and you want the output file named 'predict_summary.csv', you can use **python 03_predict_with_unlabel_data.py -i cgc_standard.out -o predict_summary.csv** where **-i** represent input and **-o** represent output. Using **python 03_predict_with_unlabel_data.py --help** for more detailed description. The results will contain three columns, the first column will be the cgc ID, the second column will be the name of the fiber substrate and the last column will be the score value which ranged from 0 to 1. The higher the score value is, the higher the possibility this cgc ID will belongs to this class. 



