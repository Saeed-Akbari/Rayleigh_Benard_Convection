import numpy as np
from tensorflow.keras import backend as K

# Custom activation (swish)
def my_swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)
