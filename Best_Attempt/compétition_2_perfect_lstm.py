# -*- coding: utf-8 -*-
"""
Compétition 2
Authors: Anshita Saxena, Denis Lemarchand, Vincent Gariépy
Team: BaudelAIre
Organization: University of Montreal (MILA)
Course: IFT 6390 (Fundamentals of Machine Learning)
"""

# Basic Libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import string

# Confusion Matrix Library
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Word processing library
from nltk.stem.snowball import SnowballStemmer
snowBallStemmer = SnowballStemmer("english")

# Tokenizers and pad_sequences libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model Dependency libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow import keras


def prepareKaggleFile(test_inputs, test_predictions, file='tests_label.csv'):
    """
    Make submission file from numpy data
    The value of class should be 0, 1 or 2 with 0 being negative, 1 being 
    neutral and 2 being positive class.
    """
    output_data_for_kaggle = np.zeros((len(test_inputs),2))
    for i in range(len(test_inputs)):
      output_data_for_kaggle[i,0] = i
      output_data_for_kaggle[i,1] = test_predictions[i]
    output_data_for_kaggle = output_data_for_kaggle.astype(int)
    print(output_data_for_kaggle)
    df = pd.DataFrame(data=output_data_for_kaggle,columns=['id','target'])
    df.to_csv(file,index=False)


def remove_class_imbalance(df_train_result):
    """
    Eradicate data Imbalance
    Ratio of 84 neutral samples to 5M positive and negative samples was discovered
    Changing the labels from neutral to positive to eradicate the data imbalance problem
    """
    df_train_result[df_train_result["target"]=='negative']=0
    df_train_result[df_train_result["target"]=='neutral']=2
    df_train_result[df_train_result["target"]=='positive']=2
    return df_train_result


def change_categorical(y_train):
    """
    This function will convert the y_test into categorical.
    """
    y_train = y_train['target']
    print(type(y_train))
    # Convert y_train into categorical column
    # This will help column to save from treating as rank 0>2
    Y_train = np.array(pd.get_dummies(y_train))
    return Y_train


def remove_punct(text):
    """
    Remove punctuation and numbers.
    """
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def stemming(text):
    """
    Snowball Stemmer is also known as the Porter2 stemming algorithm because it 
    is a better version of the Porter Stemmer. It is more aggressive than 
    Porter Stemmer.
    """
    text = [snowBallStemmer.stem(word) for word in text.split()]
    return ' '.join(text)


def preprocessing_cleanning(df_data):
      """
      This function is to preprocess the sentences based on numerous factors.
      """
      # Converting all the upper case to lower case to avoid the distinction between them
      df_data['text'] = df_data.text.str.lower()

      # Putting the regex for removing the https and www URLs
      df_data.text = df_data.text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
      df_data.text = df_data.text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))

      # Remove the video and links
      df_data.text = df_data.text.apply(lambda x: re.sub(r'{link}', '', x))
      df_data.text = df_data.text.apply(lambda x: re.sub(r"\[video\]", '', x))

      # Remove html reference characters
      df_data.text = df_data.text.apply(lambda x: re.sub(r'&[a-z]+;', '', x))

      # Remove usernames
      df_data.text = df_data.text.apply(lambda x: re.sub(r'@[^\s]+', '', x))

      # Removing numbers
      df_data.text = df_data.text.apply(lambda x: re.sub(r'\d+', '', x))

      # Removing hashmarks, non-letter characters
      df_data.text = df_data.text.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))

      # Remove punctuation
      df_data['text'] = df_data['text'].apply(lambda x: remove_punct(x))

      # Remove stop words
      stopword_list = stopwords.words('english')
      df_data['text'] = df_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword_list)]))

      # stemming
      df_data['text'] = df_data['text'].apply(lambda x: stemming(x))

      return df_data


def tokenizer_df(df_train, df_test):
    """
    This tokenizer allows to vectorize a text corpus, by turning each text into 
    either a sequence of integers (each integer being the index of a token in a 
    dictionary) or into a vector where the coefficient for each token could be 
    binary, based on word count, based on tf-idf.
    By default, lowercase is true for tokenizer.
    """
    max_features = 10000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    df_all_text = df_train['text'].append(df_test['text'])
    tokenizer.fit_on_texts(df_all_text.values)
    # Train: 1040323 rows with 116 features, i.e., (1040323, 116)
    X_train = tokenizer.texts_to_sequences(df_train['text'].values)
    X_train = pad_sequences(X_train)
    # Test: 560175 rows with 40 features, i.e., (560175, 40)
    X_test = tokenizer.texts_to_sequences(df_test['text'].values)
    X_test = pad_sequences(X_test)
    # To equalize the features of train and test set (560175, 116)
    X_test = np.lib.pad(X_test, ((0,0),(X_train.shape[1] - X_test.shape[1],0)), 
                        'constant', constant_values=(0))
    return X_train, X_test


def model(X_train):
    """
    Model creation function: LSTM network
    Max features > 10000 did not contribute in increasing the features of 
    X_train and X_test. This is the same parameter used in tokenizer.
    The embed_dim parameter is the length of the vector.
    """
    max_features = 10000
    embed_dim = 256
    model = Sequential()
    """
    An embedding layer allows us to convert each word into a fixed-length vector 
    and that is a better way to represent those words along with reduced dimensions. 
    For each word and input_length is the maximum length of a sequence.
    """
    model.add(Embedding(max_features, embed_dim,input_length = X_train.shape[1]))
    """
    The main purpose of the spatial dropout layer is to avoid overfitting and that 
    is done by probabilistically removing the inputs of this layer (or the 
    output of the embedding layer in the network we’re building).
    In the end, the nodes of the network are more robust to the future inputs and 
    tend to not overfit.
    """
    model.add(SpatialDropout1D(0.2))
    """
    The first parameter lstm_out that we’ve defined as 512 it’s the dimensionality 
    of the output space, and we can choose an even larger number trying to improve 
    our model, but that can lead to many problems like overfitting and a long 
    training time.
    The dropout parameter is applied to the inputs and/or outputs of our model 
    (the linear transformations), while the recurrent dropout is applied to the 
    recurrent state, or cell state, of the model. Recurrent dropout affects the 
    “memory” of the network. The small the dataset is, the larger the value of 
    recurrent_dropout.
    """
    lstm_out = 512
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(2,activation='sigmoid'))
    """
    An implementation of the Adam algorithm, which is a robust extended version of 
    the stochastic gradient descent.
    """
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


def model_with_epoch(X_train, Y_train, X_test, test_inputs,
                     load_model_filename, save_model_filename,
                     prediction_filename):
    """
    Load the model - We applied model each time by saving the model for 1 epoch and
    load the model for testing epoch afterwards.
    Epoch 1: Train set-80.78, Test set: 0.82497
    Epoch 2: Train set-83.12, Test set: 0.82816
    Epoch 3: Train set-84.24, Test set: 0.83065
    Epoch 4: Train set-85.13, Test set: 0.83051 (Stopped Here)
    """
    model = keras.models.load_model(load_model_filename)

    """
    The batch size is the number of samples to run through the network before a 
    weight update is performed (an epoch), we’ll keep it low as it requires less 
    memory.
    """
    batch_size = 64
    model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)

    # Save the model
    model.save(save_model_filename)

    # Predict the test set
    y_predicted = model.predict(X_test)

    # The output class is the one which will have highest probability of neuron
    pred = np.argmax(y_predicted, axis=1) * 2
    prepareKaggleFile(test_inputs, pred, file=prediction_filename)


if __name__ == '__main__':
    # Load train data into dataframe and numpy array
    df_train = pd.read_csv('../data/train.csv')
    train = df_train.to_numpy()
    train_inputs = train[:]
    print(train_inputs.shape)

    # Load train labels into dataframe and numpy array
    df_train_result = pd.read_csv('../data/train_result.csv')
    train_results = df_train_result.to_numpy()
    train_labels = train_results[:,1]
    print(train_labels.shape)

    # Load test data into dataframes and numpy array
    df_test = pd.read_csv('../data/test.csv')
    test = df_test.to_numpy()
    test_inputs = test[:]
    print(test_inputs.shape)

    # Changing the character classes to numerical classes via values to 0,1,2
    train_labels[train_labels=='negative']=0
    train_labels[train_labels=='neutral']=1
    train_labels[train_labels=='positive']=2

    # Remove imbalance
    df_train_result = remove_class_imbalance(df_train_result)
    # Change the train_labels into numerical category
    Y_train = change_categorical(df_train_result)
    # Tokenize the train and test datasets
    X_train, X_test = tokenizer_df(df_train, df_test)

    # Model Summary
    model = model(X_train)
    print(model.summary())

    # Reproducible results from Model
    np.random.seed(1234)

    """
    The batch size is the number of samples to run through the network before a 
    weight update is performed (an epoch), we’ll keep it low as it requires less 
    memory.
    """
    batch_size = 64
    model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2, shuffle=True)

    # Save the model
    filename = '../models/neural_network_lstm.h5'
    model.save(filename)

    # Predict the test set
    y_predicted = model.predict(X_test)

    # The output class is the one which will have highest probability of neuron
    """
    Source:
    # https://datascience.stackexchange.com/questions/79761/class-label-prediction-in-keras-sequential-model-showing-different-results-in-co
    # https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead
    # evidently the decided label should be the output neuron with the highest probability.
    # https://datascience.stackexchange.com/questions/36238/what-does-the-output-of-model-predict-function-from-keras-mean
    """
    pred = np.argmax(y_predicted, axis=1)*2
    prepareKaggleFile(test_inputs, pred, file='../data/test_label_nn_lstm.csv')

    model_with_epoch(X_train, Y_train, X_test, test_inputs, filename, filename,
                     '../data/test_label_nn_lstm_epoch_2.csv')

    model_with_epoch(X_train, Y_train, X_test, test_inputs, filename, filename,
                     '../data/test_label_nn_lstm_epoch_3.csv')

    model_with_epoch(X_train, Y_train, X_test, test_inputs, filename, filename,
                     '../data/test_label_nn_lstm_epoch_4.csv')
