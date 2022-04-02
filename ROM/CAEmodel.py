""" @author: Saeed  """

from preprocess import scale
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, UpSampling1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from activation import my_swish

def createCAE(lrate, latentLayer, **kwargs):
    
    encoderDense = kwargs["encoderDense"]
    decoderDense = kwargs["decoderDense"]
    decoderReshape = kwargs["decoderReshape"]
    inputShape = kwargs["inputShapeCAE"]
    encoderZipped = kwargs["encoderZipped"]
    decoderZipped = kwargs["decoderZipped"]

    
    ## Encoder
    encoder_inputs = Input(shape=inputShape,name='Field')
    enc_l = encoder_inputs
    for f, p, s in encoderZipped:
        #x = Conv2D(f,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l)
        x = Conv2D(f,kernel_size=(3,3), strides=s, activation='relu',padding='same')(enc_l)
        print(tf.shape(x))
        enc_l = MaxPooling2D(pool_size=p,padding='same')(x)
        print(tf.shape(enc_l))

    x = Flatten()(enc_l)
    print(tf.shape(x))
    for numNeuron in encoderDense:
        #x = Dense(numNeuron, activation=my_swish)(x)
        x = Dense(numNeuron)(x)
        print(tf.shape(x))
    encoded = Dense(latentLayer)(x)
    print(tf.shape(encoded))
    encoder = Model(inputs=encoder_inputs,outputs=encoded)
    
    ## Decoder
    decoder_inputs = Input(shape=(latentLayer,),name='decoded')
    print(tf.shape(decoder_inputs))
    x = decoder_inputs
    for numNeuron in decoderDense:
        #x = Dense(numNeuron, activation=my_swish)(x)
        x = Dense(numNeuron)(x)
        print(tf.shape(x))
    x = Reshape(target_shape=decoderReshape)(x)
    print(tf.shape(x))

    dec_l = x
    for f, p, s in decoderZipped:
        #x = Conv2D(f,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l)
        x = Conv2D(f,kernel_size=(3,3), strides=s, activation='relu',padding='same')(dec_l)
        print(tf.shape(x))
        dec_l = UpSampling2D(size=p)(x)
        print(tf.shape(dec_l))

    decoded = Conv2D(1,kernel_size=(3,3), strides=(1, 1),activation='linear',padding='same')(dec_l)
    print(tf.shape(decoded))

    decoder = Model(inputs=decoder_inputs,outputs=decoded)

    ## Autoencoder model
    ae_outputs = decoder(encoder(encoder_inputs))
    
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')
        
    # design network
    my_adam = optimizers.Adam(learning_rate=lrate)
    
    model.compile(optimizer=my_adam,loss='mean_squared_error', metrics=["mae"])    
    print(model.summary())

    return model, encoder, decoder


def createCAE1D(lrate, latentLayer, **kwargs):
    
    encoderDense = kwargs["encoderDense"]
    decoderDense = kwargs["decoderDense"]
    decoderReshape = kwargs["decoderReshape"]
    inputShape = kwargs["inputShapeCAE"]
    encoderZipped = kwargs["encoderZipped"]
    decoderZipped = kwargs["decoderZipped"]

    ## Encoder
    encoder_inputs = Input(shape=inputShape,name='Field')
    enc_l = encoder_inputs
    for f, p, s in encoderZipped:
        x = Conv1D(f,kernel_size=3, activation='tanh', strides=s,padding='same')(enc_l)

        #x = Dropout(0.1)(x)
        print(tf.shape(x))
        enc_l = MaxPooling1D(pool_size=p,padding='same')(x)
        print(tf.shape(enc_l))

    x = Flatten()(enc_l)
    #x = enc_l
    print(tf.shape(x))
    for numNeuron in encoderDense:
        x = Dense(numNeuron, activation='tanh')(x)
        #x = Dropout(0.1)(x)
        print(tf.shape(x))
    encoded = Dense(latentLayer, activation='tanh')(x)
    print(tf.shape(encoded))
    encoder = Model(inputs=encoder_inputs,outputs=encoded)
    
    ## Decoder
    decoder_inputs = Input(shape=(latentLayer,),name='decoded')
    x = decoder_inputs
    for numNeuron in decoderDense:
        x = Dense(numNeuron, activation='tanh')(x)
        #x = Dropout(0.1)(x)
        print(tf.shape(x))
    x = Reshape(target_shape=decoderReshape)(x)
    print(tf.shape(x))

    dec_l = x
    for f, p, s in decoderZipped:
        x = Conv1D(f,kernel_size=3, activation='tanh', strides=s,padding='same')(dec_l)
        #x = Dropout(0.1)(x)
        print(tf.shape(x))
        dec_l = UpSampling1D(size=p)(x)
        print(tf.shape(dec_l))

    decoded = Conv1D(1,kernel_size=3, strides=1,activation='tanh',padding='same')(dec_l)
    print(tf.shape(decoded))

    decoder = Model(inputs=decoder_inputs,outputs=decoded)

    ## Autoencoder model
    ae_outputs = decoder(encoder(encoder_inputs))
    
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')
        
    # design network
    my_adam = optimizers.Adam(learning_rate=lrate)
    
    model.compile(optimizer=my_adam,loss='mean_squared_error', metrics=["mae"])  # mean_squared_error  
    print(model.summary())

    return model, encoder, decoder



def createMLP(lrate, latentLayer, **kwargs):

    encoderDenseMLP = kwargs["encoderDenseMLP"]
    inputShape = kwargs["inputShapeCAE"]

    #% Encoder
    # define the input to the encoder
    encoder_inputs = Input(shape=(inputShape,))
    x = encoder_inputs
    for numNeuron in encoderDenseMLP:
        x = Dense(numNeuron,activation='tanh')(x)
    encoded = Dense(latentLayer,activation='linear')(x)
    
    encoder = Model(encoder_inputs, encoded, name="encoder")

    print(encoder.summary())        
    
    #% Decoder
    # start building the decoder model which will accept the output of the encoder
    latentInputs = Input(shape=(latentLayer,))
    
    x = latentInputs
    for numNeuron in reversed(encoderDenseMLP):
        x = Dense(numNeuron,activation='tanh')(x)

    decoded = Dense(inputShape,activation='linear')(x)
    
    decoder = Model(inputs=latentInputs, outputs=decoded, name="decoder")

    ## Autoencoder model
    ae_outputs = decoder(encoder(encoder_inputs))
    
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')
        
    # design network
    my_adam = optimizers.Adam(learning_rate=lrate)
    
    model.compile(optimizer=my_adam,loss='mean_squared_error', metrics=["mae"])    
    print(model.summary())

    return model, encoder, decoder