from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.new_constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils_seq2var import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, cutoff_sequence
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.random.set_seed(1234)
import sys

import time


# SEQ2VAR libraries
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

from sklearn.model_selection import train_test_split


# For each dataset we have to change:
# - names y methods (0,1,2)
# - DATASET_INDEX


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
args.nb_systems = X_train.shape[0]
args.timesteps =  X_train.shape[2]
args.num_atoms =  X_train.shape[1]
#args.num_atoms =  X_train.shape[1]
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
        #results = model.fit(1, trend='c')
        results = model.fit(1, trend ='n') #n, c, ct
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


# New folder for each variant of the model
names = ['VAR', 'Seq2VAR', 'bSeq2VAR']
methods = [A_vars, A_seq2vars, A_bseq2vars]

A_est = methods[2]
names = names[2]

X_train, X_test, y_train, y_test = train_test_split(A_est, l_test.numpy())
X_train = X_train.reshape(X_train.shape[0],MAX_NB_VARIABLES,MAX_NB_VARIABLES)
X_test = X_test.reshape(X_test.shape[0],MAX_NB_VARIABLES,MAX_NB_VARIABLES)
dataset2 = 'Dataset_name/'+names+'/'
path = '/home/.../TimeSeries_Classification/Review/data/'
np.save(path+dataset2+"X_train.npy", X_train)
np.save(path+dataset2+"X_test.npy", X_test)
np.save(path+dataset2+"y_train.npy", y_train)
np.save(path+dataset2+"y_test.npy", y_test)


def generate_model_2(filters1, filters2, nodes1, dropout1):
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    x = Masking()(ip)
    x = AttentionLSTM(nodes1)(x)
    x = Dropout(dropout1)(x)
    y = Permute((2, 1))(ip)
    y = Conv1D(filters1, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
    y = Conv1D(filters2, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    out = Dense(NB_CLASS, activation='softmax')(x)
    model = Model(ip, out)
    model.summary()
    # add load model code here to fine-tune
    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


df = pd.read_csv('results_train_cv.txt', sep=';', header = None)
columns = ['Dataset','fold', 'Acc', 'filters1', 'filters2','nodes1', 'dropout1']
df.columns = columns
df = df.where(df['Dataset']==DATASET_INDEX)
mean_df = df.groupby(np.arange(len(df)) // 5).mean()
selected_filters1, selected_filters2, nodes1, dropout1 = mean_df.iloc[mean_df['Acc'].idxmax(),[3,4,5,6]]

mid_time = time.time()-start_time

def main(unused_command_line_args):
    for i in range(2):
        start_time = time.time()
        model = generate_model_2(int(selected_filters1), int(selected_filters2), int(nodes1), float(dropout1))
        print('---------Modelo generado--------')
        train_model(model, DATASET_INDEX, epochs=200, batch_size=128)
        print('---------Modelo entrenado---------')
        evaluate_model(model, DATASET_INDEX, batch_size=128, startTime=start_time, mid_time=mid_time, method = names)
        print('---------Modelo evaluado----------')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))