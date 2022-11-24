import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import confusion_matrix

#This is the script that will train the best tuned model and output the predictions

#Load the encoder
word2vec_SkipGram_100d = Word2Vec.load('./Encoders/word2vec_SkipGram_100d')
word2vec_CBOW_100d = Word2Vec.load('./Encoders/word2vec_CBOW_100d')

#make submission file from numpy data
#The value of class should be 0, 1 or 2 with 0 being negative, 1 being positive
def PrepareKaggleFile(test_inputs, test_predictions, file='tests_label.csv'):
    output_data_for_kaggle = np.zeros((len(test_inputs),2))
    for i in range(len(test_inputs)):
      output_data_for_kaggle[i,0] = i
      output_data_for_kaggle[i,1] = 2 if test_predictions[i] == 1 else 0

    output_data_for_kaggle = output_data_for_kaggle.astype(int)

    print(output_data_for_kaggle)

    df = pd.DataFrame(data=output_data_for_kaggle,columns=['id','target'])
    df.to_csv(file,index=False)

#Functions to apply the word2vec encodinng to each word
def applyWord2VecCBOW_AVG(listWords):
    newList = []
    for i in listWords:
        if i in word2vec_CBOW_100d.wv:
            newList.append(word2vec_CBOW_100d.wv[i])
    return np.mean(newList,axis=0)

def applyWord2VecSG_AVG(listWords):
    newList = []
    for i in listWords:
        if i in word2vec_SkipGram_100d.wv:
            newList.append(word2vec_SkipGram_100d.wv[i])
    return np.mean(newList,axis=0)

def preprocessing(df):
    sentences = df.copy()
    # Converting all the upper case to lower case to avoid the distinction between them
    sentences['text'] = df['text'].str.lower()
    # Putting the regex for removing the https and www URLs
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))

    # Remove the video and links
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r'{link}', '', x))
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r"\[video\]", '', x))

    # Remove html reference characters
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))

    # Remove usernames
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r'@[^\s]+', '', x))

    # Removing numbers
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r'\d+', '', x))

    # Removing hashmarks, non-letter characters
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))

    # Removing all extra same letters to a limit of 2, ex. daaaang => daang, nooooooo => noo
    sentences['text'] = sentences['text'].apply(lambda x: re.sub(r"(.)\1+", r"\1\1", x))
        
    return sentences

if __name__ == '__main__':
    #Import the data
    df_train = pd.read_csv('./kaggle-competition-2/train_data.csv')
    df_test = pd.read_csv('./kaggle-competition-2/test_data.csv')
    df_train_labels = pd.read_csv('./kaggle-competition-2/train_results.csv')

    #We want to remove neutral class since there are very few of them and then we can have a binary classification model
    df_train_labels = df_train_labels[df_train_labels['target']!='neutral']
    df_train = df_train.iloc[df_train_labels.index,:]

    #Rename the labels
    df_train_labels.loc[df_train_labels['target']=='negative','target'] = 0
    df_train_labels.loc[df_train_labels['target']=='positive','target'] = 1

    #Preprocess the senteces 
    train_proc = preprocessing(df_train)
    test_proc = preprocessing(df_test)
    train_proc['text'] = train_proc['text'].apply(lambda x: x.split(' '))
    test_proc['text'] = test_proc['text'].apply(lambda x: x.split(' '))

    #Create all datasets with different encodings
    SG_Train_AVG = train_proc['text'].apply(applyWord2VecSG_AVG)
    SG_Test_AVG = test_proc['text'].apply(applyWord2VecSG_AVG)

    #Transform the data into a matrix
    SG_Train_AVG_X = pd.DataFrame(SG_Train_AVG.tolist(), index= SG_Train_AVG.index)
    SG_Test_AVG_X = pd.DataFrame(SG_Test_AVG.tolist(), index= SG_Test_AVG.index)

    #Delete variables to clear space
    del df_test, df_train
    del SG_Train_AVG, SG_Test_AVG, test_proc, train_proc, word2vec_CBOW_100d, word2vec_SkipGram_100d
    gc.collect()

    #Make sure all the data is numerical before training
    SG_train_x = SG_Train_AVG_X.apply(pd.to_numeric)
    SG_train_y = df_train_labels.apply(pd.to_numeric)

    #We can see overfitting let us add some dropout to prevent it
    model_drp = Sequential()
    model_drp.add(Dense(512, activation='relu', input_dim=100))
    model_drp.add(Dropout(0.25))
    model_drp.add(Dense(256, activation='relu'))
    model_drp.add(Dense(1, activation='sigmoid'))
    model_drp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Train on full data
    hist_SG_drp = model_drp.fit(SG_train_x, SG_train_y['target'], epochs=15, batch_size=100)

    test_pred = model_drp.predict(SG_Test_AVG_X) > 0.5

    PrepareKaggleFile(test_inputs=SG_Test_AVG_X, test_predictions=test_pred, file='tests_label_NN.csv')