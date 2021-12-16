
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence
from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score
import tensorflow as tf
from prettyprinter import pprint
import numpy as np
import pandas as pd
import argparse
import pickle

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
        #X = X.reshape(X.shape + (1,))
        y  = self.labels[list_IDs_temp]
        return X, np_utils.to_categorical(y, num_classes=self.n_classes)

def load_data(prefix):
    print("Loading dataset")
    with open(prefix + ".dat", "rb") as f:
        X = pickle.load(f)
    with open(prefix + ".lab", "rb") as f:
        Y = pickle.load(f)

    #convert labels to integers
    array_Y = np.asarray(Y)
    encoder = LabelEncoder()
    int_Y = encoder.fit_transform(array_Y)
    id_labels_dict = dict(zip(range(len(int_Y)), int_Y))
    return X, int_Y, id_labels_dict, 2, X.shape[1]

def create_model(n_classes,input_length):
    # define model
    model = Sequential()
    # The model expects rows of data with input features variables (the input_shape=n_features argument)
    # The first hidden layer has 4096 nodes and uses the relu activation function.
    model.add(Dense(4096, activation='relu', input_shape=(n_features,)))
    # The second hidden layer has 512 nodes and uses the relu activation function.
    model.add(Dense(512, activation='relu'))
    # The third hidden layer has 512 nodes and uses the relu activation function.
    model.add(Dense(512, activation='relu'))
    # The forth hidden layer has 64 nodes and uses the relu activation function.
    model.add(Dense(64, activation='relu'))
    # The output layer has 1 node and uses the sigmoid activation function.
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def train_and_evaluate_model (model, X_train, X_test, y_train, y_test, labels_dict, n_classes, n_workers, batch_size=512):
  
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    """
    training_generator = DataGenerator(X_train, range(X_train.shape[0]), y_train, batch_size=batch_size, n_classes=n_classes)

    for i in range(1,batch_size):
        if X_test.shape[0] % i == 0:
            div = i
    batch_size_test = div
    testing_generator = DataGenerator(X_test, range(X_test.shape[0]), y_test, batch_size=batch_size_test, n_classes=n_classes)
    """
    print("------------------- TRAINING ----------------------")    
    X_train = X_train.todense()
    model.fit(X_train, epochs=100, steps_per_epoch=(X_train.shape[0]/batch_size - 1), callbacks=callbacks, 
                                    workers=n_workers,
                                    use_multiprocessing=True, verbose = 1)
    tr_scores = model.evaluate(X_train, 
                            workers=n_workers,
                            use_multiprocessing=True, verbose=1)
    print("------------------- SCORES ----------------------") 
    pprint(tr_scores)

    print("------------------- TESTING ----------------------")
    X_test = X_test.todense()
    y_pred = model.predict(X_test, 
                            workers=n_workers, 
                            use_multiprocessing=True, verbose = 1) 
    #y_pred =  [0 if y[0] > y[1] else 1 for y in preds ]
    #mat = confusion_matrix(labelste, y_pred)
    te_scores = model.evaluate(X_test, 
                                workers=n_workers,
                                use_multiprocessing=True, verbose=1)
    print("------------------- SCORES ----------------------") 
    pprint(te_scores)
    return y_pred, tr_scores, te_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds and trains a CNN")
    parser.add_argument("-i", "--inprefix", required=True, help="Input prefix")
    parser.add_argument("-o", "--outprefix", required=True, help="Output prefix")
    parser.add_argument("-t", "--n_threads", help="Number of cores. Default: all available ones", type=int)
    args = parser.parse_args()

    if args.n_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)

    RANDOM_SEED = 40
    # np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # load the dataset
    X, y, labels_dict, n_classes, n_features = load_data(args.inprefix)
    
    # Split the data X,y with the train_test_split function of sklearn with parameters 
    # test_size=0.20 and random_state=RANDOM_SEED.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)

 
    # define model
    model = create_model(n_classes, n_features)

    # Perform prediction on the test data (i.e) on X_test and save the predictions in the variable y_pred.
    y_pred, tr_scores, te_scores = train_and_evaluate_model (model, X_train, X_test, y_train, y_test, labels_dict, n_classes, args.n_threads)


    
    y_pred1 = y_pred.round()
    acc = accuracy_score(y_test, y_pred1)
    print('Test Accuracy: %.3f' % acc)

    f1 = f1_score(y_test, y_pred1, average='weighted')
    print('F1 Score: %.3f ' % f1)

    recall = recall_score(y_test, y_pred1, average='weighted')
    print('Test Recall: %.3f' % recall)






