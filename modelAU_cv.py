from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.new_constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils_cv import train_model, train_selected_model, evaluate_model, set_trainable
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

import pandas as pd

DATASET_INDEX = 5

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX] 
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX] 
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX] 

TRAINABLE = True

NODES_LIST = [8, 16]  # Nodes of Attention LSTM layer
FILTERS_LIST = [64, 128, 256] # filters of Conv1D
DROPOUT_LIST = [0.25,0.5] # Dropout of the layer following the Attention LSTM
#Conv_layers = [1,2]
#EPOCHS_LIST = [50, 100, 200]
#BATCH_SIZES_LIST = [64, 128, 256]


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

def main(unused_command_line_args):

    print('---------Modelo pruebas entrenado--------')

    df = pd.read_csv('results_cv.txt', sep=';', header = None)
    columns = ['Dataset','fold', 'Acc', 'filters1', 'filters2','nodes1', 'dropout1']
    df.columns = columns
    df = df.where(df['Dataset']==4)
    mean_df = df.groupby(np.arange(len(df)) // 5).mean()
    selected_filters1, selected_filters2, nodes1, dropout1 = mean_df.iloc[mean_df['Acc'].idxmax(),[3,4,5,6]]
    
    for i in range(3):
        start_time = time.time()
        model = generate_model_2(int(selected_filters1),int(selected_filters2), int(nodes1), dropout1)
        print('---------Modelo seleccionado generado--------')
        model = train_selected_model(model, DATASET_INDEX, epochs=200, batch_size=128)
        print('---------Modelo seleccionado entreando--------')
        print('--------- ---------')
        evaluate_model(model, DATASET_INDEX, batch_size=128, startTime=start_time)
        print('---------Modelo seleccionado entrenado--------')


if __name__ == '__main__':
    sys.exit(main(sys.argv))