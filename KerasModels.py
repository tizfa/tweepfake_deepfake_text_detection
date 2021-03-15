import os

import numpy as np
import pandas as pd


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Dropout, concatenate
from keras.layers import CuDNNLSTM, GRU, Bidirectional
from keras.models import Model
import pandas as pd
from sklearn.metrics import classification_report
import pickle




def generateExpData(df, tokenizer = None, textField = "text", labelField = "account.type"):
    trainingData = df[textField].values
    trainingData = [s.lower() for s in trainingData]

    if tokenizer is None:
        # Build characters dictionary using training data.
        tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tokenizer.fit_on_texts(trainingData)


    # Convert original texts into sequences of indices.
    train_sequences = tokenizer.texts_to_sequences(trainingData)

    # Pad data to sequences of max length 320.
    train_data = pad_sequences(train_sequences, maxlen=320, padding='post')
    train_data = np.array(train_data, dtype='float32')


    return train_data, tokenizer


def buildCharCNNModel(vocabSize, embSize = 32, inputSize = 320):
    input = Input(shape=(inputSize,))
    x = Embedding(vocabSize+1, embSize)(input)
    numFilters = 128
    x3= Conv1D(numFilters, 3, activation="tanh")(x)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(numFilters, 4, activation="tanh")(x)
    x4 = GlobalMaxPooling1D()(x4)

    x5 = Conv1D(numFilters, 5, activation="tanh")(x)
    x5 = GlobalMaxPooling1D()(x5)

    conc = concatenate([x3, x4, x5])
    conc = Dropout(0.2)(conc)

    finalOut = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=input, outputs=finalOut)
    model.compile(loss=['binary_crossentropy'], optimizer='adam',
                  metrics=["accuracy"])
    print(model.summary())
    return model


def buildCharGRUModel(vocabSize, embSize = 32, inputSize = 320):

  input = Input(shape=(inputSize,))
  x = Embedding(vocabSize+1, embSize)(input)
  numFilters = 512

  gru = Bidirectional(GRU(numFilters, activation="tanh"))(x)
  conc = Dropout(0.2)(gru)

  finalOut = Dense(1, activation="sigmoid")(conc)
  model = Model(inputs=input, outputs=finalOut)
  model.compile(loss=['binary_crossentropy'], optimizer='adam',
                metrics=["accuracy"])
  print(model.summary())
  return model


def buildCharCNNAndGRUModel(vocabSize, embSize = 32, inputSize = 320):
    input = Input(shape=(inputSize,))
    x = Embedding(vocabSize+1, embSize)(input)
    numCNNFilters = 128
    numGRUFilters = 512
    x3= Conv1D(numCNNFilters, 3, activation="tanh")(x)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(numCNNFilters, 4, activation="tanh")(x)
    x4 = GlobalMaxPooling1D()(x4)

    x5 = Conv1D(numCNNFilters, 5, activation="tanh")(x)
    x5 = GlobalMaxPooling1D()(x5)

    cnn = concatenate([x3, x4, x5])
    cnn = Dropout(0.2)(cnn)

    # GRU part
    gru = Bidirectional(GRU(numGRUFilters, activation="tanh"))(x)
    gru = Dropout(0.2)(gru)

    # Concatenate
    conc = concatenate([cnn, gru])

    finalOut = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=input, outputs=finalOut)
    model.compile(loss=['binary_crossentropy'], optimizer='adam',
                  metrics=["accuracy"])
    print(model.summary())
    return model


def saveClassifierData(outputDir, model, tokenizer):
    os.makedirs(outputDir, exist_ok=True)

    # Save neural net model.
    outModelFile = outputDir + os.path.sep + "hatespeech.model"
    model.save(outModelFile)

    # Save tokenizer data.
    tokenizerFileOut = outputDir + os.path.sep+"tokenizer.pickle"
    with open(tokenizerFileOut, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

