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
import sys

import time


DATASET_INDEX = 9

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX] 
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX] 
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX] 

TRAINABLE = True


def generate_model_2():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
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
    for i in range(3):
        start_time = time.time()
        model = generate_model_2()
        print('---------Model generated--------')
        train_model(model, DATASET_INDEX, epochs=200, batch_size=128) 
        print('---------Model trained---------')
        evaluate_model(model, DATASET_INDEX, batch_size=128, startTime=start_time)
        print('---------Model evaluated----------')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))