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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#PAQUETES DE MODEL.PY
# tensorflow==2.2.0
# keras==2.3.1
# h5py==2.10.0
# numpy==1.21.0
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.new_constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, cutoff_sequence
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.random.set_seed(1234)

# Al cambiar el dataset hay que cambiar también el dataset_index en la parte de lstm

# ------------------------SEQ2VAR-------------------
#Seleccionamos dataset
path = '/home/.../TimeSeries_Classification/data/'
dataset='ECG/'
X_train = np.load(path+dataset+"X_train.npy")
labels_train = np.load(path+dataset+"y_train.npy")
X_test = np.load(path+dataset+"X_test.npy")
labels_test = np.load(path+dataset+"y_test.npy")

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
        results = model.fit(1, trend='c')
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

names = ['VAR', 'Seq2VAR', 'bSeq2VAR']

for A_est, name_method in zip([A_vars, A_seq2vars, A_bseq2vars], names):
    X_train, X_test, y_train, y_test = train_test_split(A_est, l_test.numpy())
    # Hacer CV aquí
    rf = RandomForestClassifier()
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    if name_method=='VAR':
        var = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
    elif name_method=='Seq2VAR':
        seq2var = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
    elif name_method=='bSeq2VAR':
        binary = np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test)
    print(name_method, 'Accuracy:', np.mean(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test))

executionTime = (time.time() - start_time)

file1 = open("results_seq2var_rf.txt", "a")  # append mode
file1.write(re.sub('/', '',dataset) +";" +str(var)+";" +str(seq2var)+";" +str(binary)+';'+str(executionTime)+"\n")
file1.close()

