""" @author: Saeed  """

import os
#from os import times
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocess import loadData, loadMesh, splitFlattenData, splitData,\
                        scale, transform, inverseTransform,\
                        windowDataSet, windowInverseData,\
                        windowAdapDataSet, lstmTest, probe
from visualization import animationGif, contourPlot, contourSubPlot, plot,\
                        plot1, subplot, subplotProbe, subplotMode

from LSTMmodel import createLSTM
from activation import my_swish

def main():
    
    with open('input.yaml') as file:    
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    mode = input_data['mode']
    var = input_data['var']

    PODmode = input_data['POD']['PODmode']
    px = input_data['POD']['px']
    py = input_data['POD']['py']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']
    figTimeTest = np.array(input_data['figTimeTest'])
    epochsLSTM = input_data['POD']['epochsLSTM']
    batchSizeLSTM = input_data['POD']['batchSizeLSTM']
    LrateLSTM = float(input_data['POD']['LrateLSTM'])
    validationLSTM = input_data['POD']['validationLSTM']
    windowSize = input_data['POD']['windowSize']
    timeScale = input_data['POD']['timeScale']
    numLstmNeu = input_data['POD']['numLstmNeu']
    lstm_seed = input_data['POD']['lstm_seed']
    lx = input_data['POD']['lx']
    ly = input_data['POD']['ly']

    # loading data obtained from full-order simulation
    #The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
    loc = '../FOM/'
    #flattenedPsi, flattenedTh, time, dt, mesh = loadData(loc, var)
    flattened, dt, ra, mesh = loadData(loc, var)
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime, 0.1)

    # Make a decition on which variable (temperature or stream funcion) must be trained
    if var == 'Psi':
        #flattened = np.copy(flattenedPsi)
        barRange = np.linspace(-0.60, 0.5, 30, endpoint=True)       # bar range for drawing contours
    elif var == 'Temperature':
        #flattened = np.copy(flattenedTh)
        barRange = np.linspace(0.0, 1.0, 15, endpoint=True)

    # retrieving data with its original shape (number of snapshots, first dimension of the mesh, second dimension)
    data = flattened.reshape(flattened.shape[0], (mesh[0]+1), (mesh[1]+1))
    #animationGif(Xmesh, Ymesh, data, fileName=var, figSize=(14,7))

    # Creating a directory for plots.
    dirPlot = f'plot/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)
    # Creating a directory for models.
    dirModel = f'model/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirModel):
        os.makedirs(dirModel)
    # Creating a directory for result data.
    dirResult = f'result/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirResult):
        os.makedirs(dirResult)

    # Extraction of indices for seleted times.
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0]
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0]
    testStartTime = np.argwhere(time>testStartTime)[0, 0]
    testEndTime = np.argwhere(time<testEndTime)[-1, 0]
    
    # Length of the training set
    trainDataLen = trainEndTime - trainStartTime
    # Length of the test set
    testDataLen = testEndTime - testStartTime
    
    # obtaining indices to plot the results
    for i in range(figTimeTest.shape[0]):
        figTimeTest[i] = np.argwhere(time>figTimeTest[i])[0, 0]
    figTimeTest = figTimeTest - testStartTime

    #mean subtraction
    trainStartTime = trainStartTime - 1
    flattened = flattened[trainStartTime:testEndTime].T
    flatMean = np.mean(flattened,axis=1)
    flatM = (flattened - np.expand_dims(flatMean, axis=1))

    #singular value decomposition
    Ud, Sd, _ = np.linalg.svd(flatM, full_matrices=False)
    Phid = Ud[:,:PODmode]  
    Ld = Sd**2
    #compute RIC (relative importance index)
    RICd = np.cumsum(Ld)/np.sum(Ld)*100
    #print(np.min(np.argwhere(RICd>99.99)))
    #print(np.min(np.argwhere(RICd>76.87)))

    Phid = Phid.T
    dataTest = splitData(data, data.shape[0], testStartTime, testEndTime)
    flatTestM = splitData(flatM.T, flatM.T.shape[0], testStartTime, testEndTime)

    alpha = np.dot(Phid,flatM)

    # standard scale alpha
    alpha = alpha.T
    flattenedTrain = splitData(alpha, alpha.shape[0], trainStartTime, trainEndTime)
    flattenedTest = splitData(alpha, alpha.shape[0], testStartTime, testEndTime)

    # Scale the training data
    lstmScaler = StandardScaler()
    scaledTrain = scale(flattenedTrain, lstmScaler)       # fit and transform the training set

    if mode == 'pod':

        # data shape is (number of example, features), but for POD calculations based on SVD it must be (features, number of examples)
        # after chaning the SVD to covariance matrix may not need shape change

        xtrainLSTM, ytrainLSTM = windowAdapDataSet(scaledTrain, windowSize, timeScale)

        #Shuffling data
        np.random.seed(lstm_seed)
        perm = np.random.permutation(ytrainLSTM.shape[0])
        xtrainLSTM = xtrainLSTM[perm,:,:]
        ytrainLSTM = ytrainLSTM[perm,:]

        # creating LSTM model
        lstmModel = createLSTM(LrateLSTM, PODmode, numLstmNeu)
        
        # training the model
        history = lstmModel.fit(xtrainLSTM, ytrainLSTM, epochs=epochsLSTM, batch_size=batchSizeLSTM, validation_split=validationLSTM)

        # saving the trained LSTM model
        lstmModel.save(dirModel + f'/PODlstmModel.h5')

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        plt.figure().clear()
        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = dirPlot + f'/PODlstmModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlot + f'/PODlstmModelMAE.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)
                
    elif mode == 'podTest':

        # load trained LSTM model
        lstmModel = tf.keras.models.load_model(dirModel + f'/PODlstmModel.h5')

        # predicting future data using LSTM model for test set
        scaledTest = transform(flattenedTest, lstmScaler)     # transform the test set
        ytest = lstmTest(scaledTest, lstmModel, windowSize, timeScale)

        err = np.linalg.norm(ytest - scaledTest)/np.sqrt(np.size(ytest))
        print('err = ', err)
        # inverse transform of lstm prediction before decoding
        pred = inverseTransform(ytest, lstmScaler)
        #Reconstruction
        Phid = Phid.T
        pred = pred.T 
        dataRecons = np.dot(Phid,pred)
        
        #newvar1 = (dataRecons.T + np.expand_dims(flatMean, axis=1))
        #dataTestM = newvar1.T.reshape(newvar1.T.shape[0], (mesh[0]+1), (mesh[1]+1))
        
        temp = np.expand_dims(flatMean, axis=1)

        dataRecons = (dataRecons + temp)
        reshapedData = dataRecons.T.reshape(testDataLen, (mesh[0]+1),
                                                        (mesh[1]+1))
        err = np.linalg.norm(reshapedData - dataTest)/np.sqrt(np.size(reshapedData))
        print('err = ', err)

        # plot the countor for POD-LSTM prediction for selected time
        contourSubPlot(Xmesh, Ymesh, dataTest[figTimeTest[0], :, :], reshapedData[figTimeTest[0], :, :],
                    dataTest[figTimeTest[1], :, :], reshapedData[figTimeTest[1], :, :], barRange,\
                    fileName= dirPlot + f'/PODLSTM.png', figSize=(14,7))
        
        # plotting the evolution of modes in time
        fileName = dirPlot + f'/PODlstmOutput.png'
        plotTitle = ''
        testLabel='Prediction'
        validLabel='True'
        figNum = 1
        subplot(figNum, time[testStartTime:testEndTime], scaledTest, ytest, testLabel, validLabel,\
                "Temporal snapshot", "Magnitude", plotTitle, fileName, px, py)
        '''
        # plotting the evolution of modes in time
        fileName = dirPlot + f'/lstmOutput.png'
        trainLabel='Training set'
        testLabel='Test set'
        validLabel='True data'
        figNum = 1
        aveTime = int (0.75 * (trainStartTime + trainEndTime))
        subplotMode(figNum, time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                    scaledTest, ytest, scaledSeries1[aveTime:],\
                    validLabel, testLabel, trainLabel, fileName, px, py)
        '''
        predProbe = probe(reshapedData, lx, ly, mesh[0], mesh[1])
        trueProbe = probe(dataTest, lx, ly, mesh[0], mesh[1])
        fileName = dirPlot + f'/PODprobe.png'
        plotTitle = f'evolution of {var}'
        subplotProbe(figNum, time[testStartTime:testEndTime], trueProbe, predProbe, testLabel, validLabel,\
                '', '', plotTitle, fileName, mesh, lx, ly)

        # saving the AE-LSTM prediction
        filename = dirResult + f"/PODROM"
        np.save(filename, reshapedData)
        
    elif mode == 'plot':
        '''
        # plotting the evolution of modes in time
        testLabel=''
        validLabel=''
        figNum = 1
        probeData = probe(data[trainStartTime:trainEndTime, :, :], lx, ly, mesh[0], mesh[1])
        #predProbe = probe(data[trainStartTime:trainEndTime, :, :], lx, ly, mesh[0], mesh[1])
        #trueProbe = probe(data[testStartTime:testEndTime, :, :], lx, ly, mesh[0], mesh[1])
        fileName = dirPlot + f'/probe2_{mesh[0]+1}_{mesh[1]+1}_{dt}.png'
        plotTitle = f'evolution of {var}'
        subplotProbe(time[trainStartTime:trainEndTime], probeData, '', '', plotTitle, fileName, lx, ly)
        #subplotProbe(figNum, time[trainStartTime:trainEndTime], predProbe, trueProbe, testLabel, validLabel,\
        #        '', '', plotTitle, fileName, mesh, lx, ly)
        '''
    else:
        exit()
    
if __name__ == "__main__":
    main()
