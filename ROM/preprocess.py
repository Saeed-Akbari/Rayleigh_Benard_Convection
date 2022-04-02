""" @author: Saeed  """

import os, os.path
import yaml

import numpy as np
import tensorflow as tf

# loading data of the full-order model simulation
def loadData(loc, var):

    with open(loc+'config/rbc_parameters.yaml') as file:    
        inputData = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    tmax = inputData['tmax']
    nx = inputData['nx']
    ny = inputData['ny']
    dt = inputData['dt']
    ra = float(inputData['ra'])
      
    loc = loc+'result/'
    loc = os.path.join(loc, f'solution_{tmax}_{nx}_{ny}_{dt}_{ra:0.1e}/')

    fileNames = os.listdir(loc+'save')

    fileNames = [s.replace('.npz', '') for s in fileNames]
    fileNames.sort(key = int)
    numFiles = len(fileNames)
    locations = np.empty((numFiles, 1))
    for i in range(numFiles):
        locations[i, 0] = fileNames[i]
    fileNames = [s + '.npz' for s in fileNames]

    #time = np.zeros((numFiles, 1))
    #time = np.load(loc+'time.npy')
    #time = time[1:, :]
    
    #flattenedPsi = np.empty((numFiles, (nx+1)*(ny+1)))
    #flattenedTh = np.empty((numFiles, (nx+1)*(ny+1)))

    #for i in range(numFiles):
        #temp = np.load(loc+'save/'+fileNames[i])
        #flattenedPsi[i, :] = temp.f.s.flatten()
        #flattenedTh[i, :] = temp.f.th.flatten()

    #return flattenedPsi, flattenedTh, time, dt, [nx, ny, numFiles]
    
    flattenPhi = np.empty((numFiles, (nx+1)*(ny+1)))
    if var == 'Psi':
        for i in range(numFiles):
            temp = np.load(loc+'save/'+fileNames[i])
            flattenPhi[i, :] = temp.f.s.flatten()
    elif var == 'Temperature':
        for i in range(numFiles):
            temp = np.load(loc+'save/'+fileNames[i])
            flattenPhi[i, :] = temp.f.th.flatten()
    elif var == 'Omega':
        for i in range(numFiles):
            temp = np.load(loc+'save/'+fileNames[i])
            flattenPhi[i, :] = temp.f.w.flatten()

    return flattenPhi, dt, ra, [nx, ny, numFiles]


# loading mesh
def loadMesh(loc):

    with open(loc+'config/rbc_parameters.yaml') as file:    
        inputData = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    tmax = inputData['tmax']
    nx = inputData['nx']
    ny = inputData['ny']
    dt = inputData['dt']
    ra = float(inputData['ra'])

    loc = loc+'result/'
    loc = os.path.join(loc, f'solution_{tmax}_{nx}_{ny}_{dt}_{ra:0.1e}/')
    
    Xmesh = np.empty((nx+1, ny+1))
    Ymesh = np.empty((nx+1, ny+1))

    temp = np.load(loc+f'mesh_{nx}_{ny}.npz')
    Xmesh = temp.f.X
    Ymesh = temp.f.Y

    return Xmesh, Ymesh


def splitFlattenData(data, StartTime, EndTime):

    data = data[StartTime:EndTime, :]
    return data

# splitting data to create training and test sets
def splitData(data, StartTime, EndTime):

    data = data[StartTime:EndTime]
    return data

# fit and transfom training data
def scale(data, scaler):

    data = scaler.fit_transform(data)
    return data

# transforming the test set
def transform(data, scaler):

    data = scaler.transform(data)
    return data

# inverse transform to get the original values of the data
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

def windowDataSet(training_set, lookback):
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

def windowInverseData(xtrain, ytrain):
    windowSize = np.shape(xtrain)[1]
    data = np.zeros((ytrain.shape[0]+windowSize, ytrain.shape[1]))
    data[:windowSize] = xtrain[0]
    data[windowSize:] = ytrain
    return data

def windowAdapDataSet(data, windowSize, timeScale):
    k = 0
    tmp = data[k::timeScale, :]
    xtrain, ytrain = windowDataSet(tmp, windowSize)
    for k in range(1, timeScale):
        tmp = data[k::timeScale, :]
        xtrain1, ytrain1 = windowDataSet(tmp, windowSize)
        xtrain = np.vstack([xtrain,xtrain1])
        ytrain = np.vstack([ytrain,ytrain1])
    return xtrain, ytrain

def lstmTest(data, model, ws, ts):
    ytest = np.empty_like(data)
    ytest[:ws*ts,:] = data[:ws*ts,:]
    xtest = np.copy(np.expand_dims(ytest[:ws*ts:ts,:], axis=0))
    for i in range(ws*ts, data.shape[0]):
        ytest[i,:] = model.predict(xtest) #xtest[0, -1]
        xtest = np.copy(np.expand_dims(ytest[i-ws*ts+1:i-ts+2:ts,:], axis=0))
    return ytest

def lstmTest1(data, ws):
    ytest = np.empty_like(data)
    ytest[:ws] = data[:ws]
    xtest = np.copy(np.expand_dims(ytest[:ws], axis=0))
    for i in range(ws, data.shape[0]):
        ytest[i] = xtest[0, -1] + 1
        xtest = np.copy(np.expand_dims(ytest[i-ws+1:i+1,:], axis=0))
    return ytest

def lstmTest2(data, ws):
    ytest = np.empty_like(data)
    xtest = np.zeros((1, ws, data.shape[-1]))
    xtest[0] = data[:ws]
    ytest[:ws] = data[:ws]
    ytest[ws] = xtest[0, -1] + 1
    for i in range(ws+1, ytest.shape[0]):
        xtest[0] = np.roll(xtest[0], -1)
        xtest[0, ws-1] = ytest[i-1]
        ytest[i] = xtest[0, ws-1] + 1
    return ytest

'''
def spatSVD(x):
    w, v = np.linalg.eig(x@x.T)
    m = np.shape(x)[0]
    v = v[:, :m]
    w = np.sqrt(w[:m])

    return v, w
'''

def probe(phi, lx, ly, nx, ny):
    probe = np.zeros((phi.shape[0], len(lx), len(ly)))
    for i in range(len(lx)):
        for j in range(len(ly)):
            probe[:, i, j] = phi[:, int(lx[i]*nx/8), int(ly[j]*ny/4)]

    return probe
