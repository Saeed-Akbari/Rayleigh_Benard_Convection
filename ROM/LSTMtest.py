""" @author: Saeed  """

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

def main():

    series = np.array(range(80))
    series2D = np.array(range(80*2)).reshape(80, 2)
    series3D = np.array(range(80*2*3)).reshape(80, 2, 3)
    

    split_time = 60
    x_train = series[:split_time]
    x_valid = series[split_time:]
    x_train2D = series2D[:split_time, :]
    x_valid2D = series2D[split_time:, :]
    x_train3D = series3D[:split_time, :, :]
    x_valid3D = series3D[split_time:, :, :]

    window_size = 20
    batch_size = 8
    shuffle_buffer_size = 1000

    #dataset = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    dataset2D = windowed_dataset(x_train2D, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    #dataset3D = windowed_dataset(x_train3D, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    #print(list(dataset2D.as_numpy_iterator()))
    ''' 
    print("###")
    #print(list(dataset2D.as_numpy_iterator()))
    for window in dataset2D:
        print(list(window.as_numpy_iterator()))
    print("###")
    
    print("###")
    print(list(dataset.as_numpy_iterator()))
    #for window in dataset:
        #print(list(window.as_numpy_iterator()))
    print("###")
    
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    print(list(dataset.as_numpy_iterator()))

    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    print(list(dataset.as_numpy_iterator()))
    '''


    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=[None, 1]),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 200)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    history = model.fit(dataset,epochs=100)
    '''
if __name__ == "__main__":
    main()
