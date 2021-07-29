""" @author: Saeed  """

import os, os.path
import yaml

import numpy as np
import tensorflow as tf

def loadData(loc):

    with open(loc+'config/rbc_parameters.yaml') as file:    
        inputData = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    nx = inputData['nx']
    ny = inputData['ny']
    ra = float(inputData['ra'])

    loc = loc+'RBC_FST_3/'
    loc = os.path.join(loc, f"solution_{nx}_{ny}_{ra:0.1e}_2_3_1/")

    fileNames = os.listdir(loc+'save')
    fileNames = [s.replace('.npz', '') for s in fileNames]
    fileNames.sort(key = int)
    numFiles = len(fileNames)
    locations = np.empty((numFiles, 1))
    for i in range(numFiles):
        locations[i, 0] = fileNames[i]
    fileNames = [s + '.npz' for s in fileNames]

    time = np.zeros((numFiles, 1))
    time = np.load(loc+'time.npy')
    time = time[1:, :]
    
    dataW = np.empty((numFiles, nx+1, ny+1))
    dataPsi = np.empty((numFiles, nx+1, ny+1))
    dataTh = np.empty((numFiles, nx+1, ny+1))
    flattenedW = np.empty((numFiles, (nx+1)*(ny+1)))
    flattenedPsi = np.empty((numFiles, (nx+1)*(ny+1)))
    flattenedTh = np.empty((numFiles, (nx+1)*(ny+1)))

    for i in range(numFiles):
        temp = np.load(loc+'save/'+fileNames[i])
        dataW[i, :, :] = temp.f.w
        dataPsi[i, :, :] = temp.f.s
        dataTh[i, :, :] = temp.f.th
        flattenedW[i, :] = temp.f.w.flatten()
        flattenedPsi[i, :] = temp.f.s.flatten()
        flattenedTh[i, :] = temp.f.th.flatten()

    return dataPsi, dataTh, flattenedPsi, flattenedTh, time, [nx, ny, numFiles]

def loadMesh(loc):

    with open(loc+'config/rbc_parameters.yaml') as file:    
        inputData = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    nx = inputData['nx']
    ny = inputData['ny']
    ra = float(inputData['ra'])

    loc = loc+'RBC_FST_3/'
    loc = os.path.join(loc, f"solution_{nx}_{ny}_{ra:0.1e}_2_3_1/")

    Xmesh = np.empty((nx+1, ny+1))
    Ymesh = np.empty((nx+1, ny+1))

    temp = np.load(loc+f'mesh_{nx}_{ny}.npz')
    Xmesh = temp.f.X
    Ymesh = temp.f.Y

    return Xmesh, Ymesh


def splitFlattenData(data, numExample, StartTime, EndTime):

    data = data[StartTime:EndTime, :]
    return data

def splitData(data, numExample, StartTime, EndTime):

    data = data[StartTime:EndTime, :, :]
    return data

def scale(data, scaler):

    data = scaler.fit_transform(data)
    return data

def transform(data, scaler):

    data = scaler.transform(data)
    return data

def inverseTransform(data, scaler):

    data = scaler.inverse_transform(data)
    return data

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

def createLSTMdata(training_set, lookback):
    m = training_set.shape[0]
    n = training_set.shape[1]
    ytrain = [training_set[i+1] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a

    return xtrain, ytrain

def createLSTMdata1(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain