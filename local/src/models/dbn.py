#TODO this module does not work because the classes defined for the DBN should be adapted to the current data types

import numpy as np
import sys
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import pandas as pd
import argparse
import pickle

from dbn import SupervisedDBNClassification

#parameters: sys.argv[1] = input dataset as matrix of k-mers

# Loading dataset

"""
nome_train=sys.argv[1].split(".")[0]				
def load_data(file):
	lista=[]
	records= list(open(file, "r"))
	for seq in records:
		elements=seq.split(",")
		level=elements[-1].split("\n")
		classe=level[0]
		lista.append(classe)

	lista=set(lista)
	classes=list(lista)
	X=[]
	Y=[]
	for seq in records:
		elements=seq.split(",")
		X.append(elements[1:-1])
		level=elements[-1].split("\n")
		classe=level[0]
		Y.append(classes.index(classe))
	X=np.array(X,dtype=float)
	Y=np.array(Y,dtype=int)
	data_max= np.amax(X)
	X = X/data_max
	return X,Y,len(classes),len(X[0])
"""

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
    #convert scipy sparse matrix to pandas sparse dataframe
    #df_X = pd.DataFrame.sparse.from_spmatrix(X)
    #convert labels to integers
    array_Y = np.asarray(Y)
    encoder = LabelEncoder()
    int_Y = encoder.fit_transform(array_Y)
    id_labels_dict = dict(zip(range(len(int_Y)), int_Y))
    return X, int_Y, id_labels_dict, 2, X.shape[1]


# Training
def create_model():
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=512,
                                             activation_function='relu',
                                             dropout_p=0.2,verbose=True)
    return classifier

def train_and_evaluate_model (model, X_train, X_test, labels_dict, y_train, y_test, n_classes, nb_workers, batch_size=512):
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

    training_generator = DataGenerator(X_train, range(X_train.shape[0]), y_train, batch_size=batch_size, n_classes=n_classes)

    for i in range(1,batch_size):
        if X_test.shape[0] % i == 0:
            div = i
    batch_size_test = div   
    testing_generator = DataGenerator(X_test, range(X_test.shape[0]), y_test, batch_size=batch_size_test, n_classes=n_classes)
    print("------------------- TRAINING ----------------------")
    model.fit(training_generator)

    print("------------------- TESTING ----------------------")
    y_pred = model.predict(testing_generator)

    
    acc = accuracy_score(y_test, y_pred)
    return y_pred, y_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds and trains a CNN")
    parser.add_argument("-i", "--inprefix", required=True, help="Input prefix")
    parser.add_argument("-o", "--outprefix", required=True, help="Output prefix")
    parser.add_argument("-t", "--n_threads", help="Number of cores. Default: all available ones", type=int)
    args = parser.parse_args()

    if args.n_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)
    
    
    X,Y, labels_dict, n_classes, n_features = load_data(args.inprefix)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    model = create_model()
    y_pred, y_test  = train_and_evaluate_model(model, X_train, X_test, labels_dict, y_train, y_test, n_classes, args.n_threads)

    np.save(args.outprefix+"_preds",y_pred)
    np.save(args.outprefix+"_test",y_test)
    
    #np.save(args.outprefix+"_cm",cm)

    #plot_confusion_matrix(cm, args.outprefix)
    
    model.save(args.outprefix + "_cnn.mdl")
    
    """
    n_folds = 10
    X_train,Y_train,nb_classes,input_length = load_data(sys.argv[1])
    i=1
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for train, test in kfold.split(X_train, Y_train):
        model = None # Clearing the NN.
        model = create_model()
        pred, Y_test = train_and_evaluate_model(model, X_train[train], Y_train[train], X_train[test], Y_train[test])
        np.save("./results/preds_"+nome_train+"_"+str(i),pred)
        np.save("./results/test_"+nome_train+"_"+str(i),Y_test)
        i = i+1
    """
