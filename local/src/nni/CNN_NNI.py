from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.metrics import Precision, Recall
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import argparse
import pickle
import seaborn as sns
import matplotlib.pyplot as plt     
import nni
import logging

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("xenominer")



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Custom generator to load data progressively and avoid, overflowing RAM

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, list_IDs, labels, batch_size=32, 
                 n_classes=2, shuffle=True):
        'Initialization'
        self.X = X
        self.batch_size = batch_size
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
        # Generate data
        # Store sample
        X = np.asarray(self.X[list_IDs_temp, :].todense())
        X = X.reshape(X.shape + (1,))
        y  = self.labels[list_IDs_temp]
        return X, np_utils.to_categorical(y, num_classes=self.n_classes)


def load_data(prefix):
    print("Loading dataset")
    with open(prefix + ".dat", "rb") as f:
        X = pickle.load(f)
    with open(prefix + ".lab", "rb") as f:
        Y = pickle.load(f)
    #convert scipy sparse matrix to pandas sparse dataframe
    #df_X = pd.DataFrame.sparse.from_spmatrix(X)
    #convert labels to integers
    array_Y = np.asarray(Y)
    encoder = LabelEncoder()
    int_Y = encoder.fit_transform(array_Y)
    id_labels_dict = dict(zip(range(len(int_Y)), int_Y))
    return X, int_Y, id_labels_dict, 2, X.shape[1]

def create_model(nb_classes,input_length, params):
    model = Sequential()
    model.add(Convolution1D(params["kernel_size1"],params["kernel_n1"], padding='valid', input_shape=(input_length, 1))) #input_dim
    model.add(Activation(params["activation_function1"]))
    model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Convolution1D(params["kernel_size2"],params["kernel_n2"],padding='valid'))
    model.add(Activation(params["activation_function2"]))
    model.add(MaxPooling1D(pool_size=2,padding='valid'))
    model.add(Flatten())
    ##
    ##MLP
    model.add(Dense(500))
    model.add(Activation(params["activation_function3"]))
    model.add(Dropout(params["dropout_rate"]))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    optimizer = Adam(lr=params["learning_rate"])
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy', #previous: categorical_crossentropy
              metrics=['accuracy', f1_m, Precision(), Recall()])
    print(model.summary())
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


def train_and_evaluate(args, params):

    # Load data
    X,Y, labels_dict, nb_classes,input_length = load_data(args.inprefix)
    model = create_model(nb_classes,input_length, params)

    # Split train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    training_generator = DataGenerator(X_train, range(X_train.shape[0]),y_train, batch_size=params["batch_size"], n_classes=nb_classes)

    for i in range(1,params["batch_size"]):
        if X_test.shape[0] % i == 0:
            div = i
    batch_size_test = div
    testing_generator = DataGenerator(X_test, range(X_test.shape[0]), y_test, batch_size=batch_size_test, n_classes=nb_classes)
 
    #train
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    model.fit(training_generator, epochs=100, steps_per_epoch=X_train.shape[0]/params["batch_size"], callbacks=callbacks, 
                                    workers=args.n_threads,
                                    use_multiprocessing=True, verbose = 1)
    tr_scores = model.evaluate(training_generator, 
                            workers=args.n_threads,
                            use_multiprocessing=True, verbose=1)
                            

    LOG.debug('Training final result is: loss = %d, accuracy = %d, f1 = %d, precision = %d, recall = %d', tr_scores[0], tr_scores[1], tr_scores[2], tr_scores[3], tr_scores[4])
    #nni.report_final_result(precision)


    #test
    preds = model.predict(testing_generator, workers=args.n_threads, 
                            use_multiprocessing=True, verbose = 1) 
    te_scores = model.evaluate(testing_generator, 
                                workers=args.n_threads,
                                use_multiprocessing=True, verbose=1)
    LOG.debug('Tesing final result is: loss = %d, accuracy = %d, f1 = %d, precision = %d, recall = %d', te_scores[0], te_scores[1], te_scores[2], te_scores[3], te_scores[4])
    nni.report_final_result(te_scores[4]) #recall


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds and trains a CNN")
    parser.add_argument("-i", "--inprefix", required=True, help="Input prefix")
    #parser.add_argument("-o", "--outprefix", required=True, help="Output prefix")
    parser.add_argument("-t", "--n_threads", help="Number of cores. Default: all available ones", type=int)
    args = parser.parse_args()

    if args.n_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)
    
    """
    n_folds = 10
    X,Y, labels_dict, nb_classes,input_length = load_data(args.inprefix)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    model = create_model(nb_classes,input_length)
    pred, Y_test, tr_scores, te_scores, cm = train_and_evaluate_model(model, X_train, X_test, labels_dict, y_train, y_test, nb_classes, args.n_threads)

    np.save(args.outprefix+"_preds",pred)
    np.save(args.outprefix+"_test",y_test)
    np.save(args.outprefix+"_tr_scores",tr_scores)
    np.save(args.outprefix+"_te_scores",te_scores)
    #np.save(args.outprefix+"_cm",cm)

    plot_confusion_matrix(cm, args.outprefix)

    model.save(args.outprefix + "_cnn.mdl")
    #model = keras.models.load_model('path/to/location')
    """
    try:
        # get parameters from tuner
        # RECEIVED_PARAMS = {"optimizer": "Adam", "learning_rate": 0.00001}
        PARAMS = nni.get_next_parameter()
        LOG.debug(PARAMS)
        # train
        train_and_evaluate(args, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
    