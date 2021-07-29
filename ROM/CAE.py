""" @author: Saeed  """

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from activation import my_swish

def createCAE(lrate, latentLayer, **kwargs):
    
    encoderFilters = kwargs["encoderFilters"]
    encoderDense = kwargs["encoderDense"]
    decoderFilters = kwargs["decoderFilters"]
    decoderDense = kwargs["decoderDense"]
    decoderReshape = kwargs["decoderReshape"]
    inputShape = kwargs["inputShapeCAE"]
    
    ## Encoder
    encoder_inputs = Input(shape=inputShape,name='Field')
    enc_l = encoder_inputs
    for f in encoderFilters:
        x = Conv2D(f,kernel_size=(3,3),activation=my_swish,padding='same')(enc_l)
        print(tf.shape(x))
        enc_l = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
        print(tf.shape(enc_l))

    x = Flatten()(enc_l)
    print(tf.shape(x))
    for numNeuron in encoderDense:
        x = Dense(numNeuron, activation=my_swish)(x)
        print(tf.shape(x))
    encoded = Dense(latentLayer)(x)
    print(tf.shape(encoded))
    encoder = Model(inputs=encoder_inputs,outputs=encoded)
    
    ## Decoder
    decoder_inputs = Input(shape=(latentLayer,),name='decoded')
    print(tf.shape(decoder_inputs))
    x = decoder_inputs
    for numNeuron in decoderDense:
        x = Dense(numNeuron, activation=my_swish)(x)
        print(tf.shape(x))
    x = Reshape(target_shape=decoderReshape)(x)
    print(tf.shape(x))

    dec_l = x

    for f in decoderFilters:
        x = Conv2D(f,kernel_size=(3,3),activation=my_swish,padding='same')(dec_l)
        print(tf.shape(x))
        dec_l = UpSampling2D(size=(2, 2))(x)
        print(tf.shape(dec_l))

    decoded = Conv2D(1,kernel_size=(3,3),activation='linear',padding='same')(dec_l)
    print(tf.shape(decoded))

    decoder = Model(inputs=decoder_inputs,outputs=decoded)

    ## Autoencoder
    ae_outputs = decoder(encoder(encoder_inputs))
    
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')
        
    # design network
    my_adam = optimizers.Adam(learning_rate=lrate)
    
    model.compile(optimizer=my_adam,loss='mean_squared_error', metrics=["mae"])    
    #model.summary()

    return model, encoder, decoder

