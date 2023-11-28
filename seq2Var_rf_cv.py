# PAQUETES DE SEQ2VAR
import copy
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from utils.seq2Var.utils import *
from utils.seq2Var.modules import *
from utils.seq2Var.data_generator_permutations import *
from utils.seq2Var.argument_parser_permutations import *

from itertools import permutations
from statsmodels.tsa.vector_ar.var_model import VAR

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#PAQUETES DE MODEL.PY
# tensorflow==2.2.0
# keras==2.3.1
# h5py==2.10.0
# numpy==1.21.0
# import keras
# from keras.models import Model
# from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
# from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.new_constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST

from utils.generic_utils import load_dataset_at
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.random.set_seed(1234)


# When changin dataset we have to also change the dataset_index in the lstm part
# ------------------------CÓDIGO SEQ2VAR-------------------
#Select dataset
DATASET_INDEX = 0
MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX] 
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX] 
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX] 

X_train, labels_train, X_test, labels_test, is_timeseries = load_dataset_at(DATASET_INDEX,
                                                                fold_index=None,
                                                                normalize_timeseries=False)


X_train = torch.Tensor(np.expand_dims(X_train,-1))
labels_train = torch.Tensor(labels_train.ravel())
X_test = torch.Tensor(np.expand_dims(X_test,-1))
labels_test = torch.Tensor(labels_test.ravel())

print(X_train.shape)
print(labels_train.shape)
print(X_test.shape)
print(labels_test.shape)

args = argument_parser()
args.sd = 0
args.nb_systems = X_train.shape[1]
args.timesteps =  X_train.shape[2]
args.num_atoms =  X_train.shape[1]
print(args)

train_data = TensorDataset(X_train, labels_train)
train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

test_data = TensorDataset(X_test, labels_test)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Training the different models

### Seq2VAR: relational encoder with linear decoder for MTS representation learning
#for i in range(5):
start_time = time.time()

args_seq2var = argument_parser_seq2var()
args_seq2var.epochs = 200
print(args_seq2var)

encoder_seq2var = RelationalEncoder(args.timesteps, args_seq2var.encoder_hidden, args.lag)

if args.cuda:
    encoder_seq2var = encoder_seq2var.cuda()

optimizer_encoder = optim.Adam(encoder_seq2var.parameters(), lr=args_seq2var.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=250, gamma=0.5)

train_seq2var(train_data_loader, encoder_seq2var, optimizer_encoder, scheduler, args, args_seq2var, binary=False)


### Binary Seq2VAR

off_diag = np.ones([args.num_atoms, args.num_atoms])

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

encoder_bseq2var = RelationalEncoder(args.timesteps, args_seq2var.encoder_hidden, args.lag)

if args.cuda:
    encoder_bseq2var = encoder_bseq2var.cuda()

optimizer_encoder = optim.Adam(encoder_bseq2var.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=250, gamma=0.5)

train_seq2var(train_data_loader, encoder_bseq2var, optimizer_encoder, scheduler, args, args_seq2var, binary=True)



##PERFORMANCES

l_test, A_vars, A_seq2vars, A_bseq2vars, A_nris, A_GT = [], [], [], [], [], []
encoder_seq2var.eval()
encoder_bseq2var.eval()
# nri_encoder.eval()
# nri_decoder.eval()

for data, label in test_data_loader:

    l_test.append(label)
    X = data

    # VAR
    for d in data:
        model = VAR(d.squeeze().numpy().T)
        results = model.fit(1, trend='n')
        A_vars.append(results.coefs)

    # seq2VAR
    off_diag = np.ones([args.num_atoms, args.num_atoms])
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    A_seq2var = encoder_seq2var.cpu()(X, rel_rec, rel_send)
    A_seq2vars.append(A_seq2var.contiguous().view(-1, args.num_atoms, args.num_atoms).cpu().detach())

    # binary seq2VAR
    A_bseq2VAR = encoder_bseq2var.cpu()(X, rel_rec, rel_send)
    A_bseq2VAR = (A_bseq2VAR > 0).float()
    A_bseq2vars.append(A_bseq2VAR.contiguous().view(-1, args.num_atoms, args.num_atoms).cpu().detach())



A_vars_binarized = torch.FloatTensor(np.concatenate([(np.abs(w) > np.percentile(np.abs(w), q=90)) for w in A_vars]).astype(float))
A_vars = torch.FloatTensor(np.concatenate(A_vars))
A_seq2vars = torch.cat(A_seq2vars)
A_seq2vars_binarized = torch.stack([(w.abs() > np.percentile(w.abs(), q=90)) for w in A_seq2vars]).float()
A_bseq2vars = torch.cat(A_bseq2vars)
# A_nris = torch.transpose(torch.cat(A_nris), 1, 2)
# A_GT = torch.FloatTensor(np.concatenate(A_GT))
l_test = torch.cat(l_test)



#-------------------------RF GRID-----------------------
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 100, num = 5)]  # Número de trees en cada forest
max_features = ['log2','sqrt',1.0] # Number of features to consider at every split (sqrt, log, None)
max_depth = [int(x) for x in np.linspace(5, 15, num = 2)]  # Maximum number of levels in tree
max_depth.append(None) # Minimum number of samples required to split a node
# min_samples_split = [2,4]  # Minimum number of samples required at each leaf node
# min_samples_leaf = [2,4]   # Method of selecting samples for training each tree
bootstrap = [True]# Create the random grid

rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

params = rf_grid
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = params, cv = 5, verbose = 1)


mid_time = (time.time() - start_time)

names = ['VAR', 'Seq2VAR', 'bSeq2VAR']

for A_est, name_method in zip([A_vars, A_seq2vars, A_bseq2vars], names):
    X_train, X_test, y_train, y_test = train_test_split(A_est, l_test.numpy())
    grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)

for i in range(3):
    for A_est, name_method in zip([A_vars, A_seq2vars, A_bseq2vars], names):   
        X_train, X_test, y_train, y_test = train_test_split(A_est, l_test.numpy())
        start_time = time.time()
        
        rf = RandomForestClassifier(**grid_search.best_params_)
        rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

        if name_method=='VAR':
            var = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
        elif name_method=='Seq2VAR':
            seq2var = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
        elif name_method=='bSeq2VAR':
            binary = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
        print(name_method, 'Accuracy:', np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test))
        print(grid_search.best_params_)
        params_row = grid_search.best_params_['bootstrap'], grid_search.best_params_['max_depth'], grid_search.best_params_['max_features'], grid_search.best_params_['n_estimators']


    # Execution time excluding the time it takes to search for hyperparameters
    executionTime = (time.time() - start_time) + mid_time

    file1 = open("results_seq2var_rf_cv.txt", "a")  # append mode
    file1.write(re.sub('/', '',str(DATASET_INDEX)) +";" +str(var)+";" +str(seq2var)+";" +str(binary)+';'+str(executionTime)+';'+str(params_row)+"\n")
    file1.close()

