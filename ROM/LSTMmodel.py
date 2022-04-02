""" @author: Saeed  """

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def createLSTM(lrate = 0.001, num_latent=8, *args):
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, return_sequences=True),
    #tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(10, return_sequences=False),
    tf.keras.layers.Dense(num_latent),
    ])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=["mae"])

    return model
