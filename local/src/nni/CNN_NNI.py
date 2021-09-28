from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import pandas as pd
import argparse
import pickle

import nni

# Custom generator to load data progressively and avoid, overflowing RAM

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, df, list_IDs, labels, batch_size=32, input_length=32,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.df = df
        self.batch_size = batch_size
        self.input_length = input_length
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.input_length), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        # Store sample
        X = self.df.loc[list_IDs_temp]

        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]

        X = X.values.reshape(X.shape + (1,))
        return X, np_utils.to_categorical(y, num_classes=self.n_classes)


def load_data(prefix):
    #TODO convert sparse matrices in sparse pandas dataframe
    print("Loading dataset")
    with open(prefix + ".dat", "rb") as f:
        X = pickle.load(f)
    with open(prefix + ".lab", "rb") as f:
        Y = pickle.load(f)
    #convert scipy sparse matrix to pandas sparse dataframe
    print("Converting to pandas sparse dataframe")
    df_X = pd.DataFrame.sparse.from_spmatrix(X)
    #convert labels to integers
    print("Converting labels to integer labels")
    array_Y = np.asarray(Y)
    encoder = LabelEncoder()
    int_Y = encoder.fit_transform(array_Y)
    id_labels_dict = dict(zip(df_X.index.tolist(), int_Y))
    return df_X,int_Y, id_labels_dict, 2, X.shape[1]

def create_model(nb_classes,input_length):
    model = Sequential()
    model.add(Convolution1D(5,5, padding='valid',  input_shape=(input_length, 1))) #input_dim
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Convolution1D(10, 5,padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Flatten())
    ##
    ##MLP
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

#Nested k fold cross validation
#https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
# example: k-fold cross validation for hyperparameter optimization (k=3)

#original data split into training and test set:

#|---------------- train ---------------------|         |--- test ---|

#cross-validation: test set is not used, error is calculated from
#validation set (k-times) and averaged:

#|---- train ------------------|- validation -|         |--- test ---|
#|---- train ---|- validation -|---- train ---|         |--- test ---|
#|- validation -|----------- train -----------|         |--- test ---|

#final measure of model performance: model is trained on all training data
#and the error is calculated from test set:

#|---------------- train ---------------------|--- test ---|


def train_and_evaluate_model (model, datatr, datate, labels_dict, labelstr, labelste, input_length, nb_classes, nb_workers):

    #TODO this line transform the dataset to 3D dataset, how to do it with pandas?
    #datatr = datatr.values.reshape(datatr.shape + (1,))
    labelstr = np_utils.to_categorical(labelstr, nb_classes)
    labelste_bin = np_utils.to_categorical(labelste, nb_classes)
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    training_generator = DataGenerator(datatr, datatr.index.tolist(), labels_dict, batch_size=64, input_length=input_length, n_classes=nb_classes)
    testing_generator = DataGenerator(datate, datate.index.tolist(),  labels_dict, batch_size=64, input_length=input_length, n_classes=nb_classes)
    print("------------------- TRAINING ----------------------")
    #model.fit(training_generator,validation_data=validation_generator,epochs=100,workers=nb_workers,callbacks = callbacks,verbose=1)
    
    model.fit(training_generator, epochs=100, callbacks=callbacks, workers=nb_workers,use_multiprocessing=True, verbose = 1)
    tr_scores = model.evaluate(datatr,labelstr,verbose=1)

    #datate = datate.values.reshape(datate.shape + (1,))
    #print(tr_scores)

    print("------------------- TESTING ----------------------")
    preds = model.predict(testing_generator, workers=nb_workers, use_multiprocessing=False, verbose = 1) 
    te_scores = model.evaluate(datate, labelste_bin,verbose=1)
    return preds, labelste, tr_scores, te_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds and trains a CNN")
    parser.add_argument("-i", "--inprefix", required=True, help="Input prefix")
    parser.add_argument("-o", "--outprefix", required=True, help="Output prefix")
    parser.add_argument("-t", "--n_threads", help="Number of cores. Default: all available ones", type=int)
    args = parser.parse_args()
    
    n_folds = 10
    X,Y, labels_dict, nb_classes,input_length = load_data(args.inprefix)
    i=1
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
    if args.n_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)
    for train, test in kfold.split(X, Y):
        print("Running fold {}".format(i))
        model = None # Clearing the NN.
        model = create_model(nb_classes,input_length)
        pred,Y_test, tr_scores, te_scores = train_and_evaluate_model(model, X.iloc[train], X.iloc[test], labels_dict, Y[train], Y[test], input_length, nb_classes, args.n_threads)
        np.save(args.outprefix+"_preds"+str(i),pred)
        np.save(args.outprefix+"_test"+str(i),Y[test])
        np.save(args.outprefix+"_tr_scores"+str(i),tr_scores)
        np.save(args.outprefix+"_te_scores"+str(i),te_scores)
        i=i+1