""" @author: Saeed  """

import os
#from os import times
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocess import loadData, loadMesh, splitFlattenData, splitData,\
                        scale, transform, inverseTransform,\
                        windowDataSet, windowInverseData,\
                        windowAdapDataSet, lstmTest, probe
from visualization import animationGif, contourPlot, contourSubPlot, plot,\
                        plot1, subplotProbe, subplotMode, plotPODcontent

from CAEmodel import createCAE1D, createMLP
from LSTMmodel import createLSTM
from activation import my_swish

def main():
    
    with open('input.yaml') as file:    
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    mode = input_data['mode']
    var = input_data['var']
    
    podContent = input_data['POD-AE']['podContent']
    epochsCAE = input_data['POD-AE']['epochsCAE']
    batchSizeCAE = input_data['POD-AE']['batchSizeCAE']
    LrateCAE = float(input_data['POD-AE']['LrateCAE'])
    AEmode = input_data['POD-AE']['AEmode']
    #PODmode = input_data['POD-AE']['PODmode']
    px = input_data['POD-AE']['px']
    py = input_data['POD-AE']['py']
    numChannel = input_data['POD-AE']['numChannel']
    sn = input_data['POD-AE']['sn']
    cae_seed = input_data['POD-AE']['cae_seed']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']
    figTimeTest = np.array(input_data['figTimeTest'])
    timeStep = np.array(input_data['timeStep'])
    epochsLSTM = input_data['POD-AE']['epochsLSTM']
    batchSizeLSTM = input_data['POD-AE']['batchSizeLSTM']
    LrateLSTM = float(input_data['POD-AE']['LrateLSTM'])
    validationLSTM = input_data['POD-AE']['validationLSTM']
    windowSize = input_data['POD-AE']['windowSize']
    timeScale = input_data['POD-AE']['timeScale']
    numLstmNeu = input_data['POD-AE']['numLstmNeu']
    lstm_seed = input_data['POD-AE']['lstm_seed']
    lx = input_data['POD-AE']['lx']
    ly = input_data['POD-AE']['ly']
    
    encoderDenseMLP = input_data['POD-AE']['encoderDenseMLP']

    encoderFilters = input_data['POD-AE']['encoderFilters']
    encoderPoolSize = input_data['POD-AE']['encoderPoolSize']
    encoderStride = input_data['POD-AE']['encoderStride']
    encoderDense = input_data['POD-AE']['encoderDense']
    decoderDense = input_data['POD-AE']['decoderDense']
    decoderReshape = input_data['POD-AE']['decoderReshape']
    decoderFilters = input_data['POD-AE']['decoderFilters']
    decoderPoolSize = input_data['POD-AE']['decoderPoolSize']
    decoderStride = input_data['POD-AE']['decoderStride']

    encoderZipped = zip(encoderFilters, encoderPoolSize, encoderStride)
    decoderZipped = zip(decoderFilters, decoderPoolSize, decoderStride)
    
    # loading data obtained from full-order simulation
    #The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
    loc = '../FOM/'
    #flattenedPsi, flattenedTh, time, dt, mesh = loadData(loc, var)
    flattened, dt, ra, mesh = loadData(loc, var)
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime, timeStep)

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
    dirPlot = f'plot/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)
    # Creating a directory for models.
    dirModel = f'model/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirModel):
        os.makedirs(dirModel)
    # Creating a directory for result data.
    dirResult = f'result/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
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

    # data splitting
    dataTest = splitData(data, testStartTime, testEndTime)
    flatMTrain = splitData(flatM.T, trainStartTime, trainEndTime).T
    flatMTest = splitData(flatM.T, testStartTime, testEndTime).T

    #singular value decomposition
    #Ud, Sd, _ = np.linalg.svd(flatM, full_matrices=False)
    Ud, Sd, _ = np.linalg.svd(flatMTrain, full_matrices=False)
    #compute RIC (relative importance index)
    Ld = Sd**2
    RICd = np.cumsum(Ld)/np.sum(Ld)*100
    PODmode = np.min(np.argwhere(RICd>99.9))

    # Convolutional autoencoder input shape
    inputShapeCAE = (PODmode, numChannel)
    
    PhidTrain = Ud[:,:PODmode]
    PhidTrain = PhidTrain.T
    alphaTrain = np.dot(PhidTrain,flatMTrain)
    alphaTest = np.dot(PhidTrain,flatMTest)

    if podContent:
        plotPODcontent(RICd, AEmode, dirPlot)

    # standard scale alpha
    alphaTrain = alphaTrain.T
    alphaTest = alphaTest.T
    #flattenedTrain = splitData(alpha, alpha.shape[0], trainStartTime, trainEndTime)
    #flattenedTest = splitData(alpha, alpha.shape[0], testStartTime, testEndTime)

    # Scale the training data
    aeScaler = MinMaxScaler()
    xtrain = scale(alphaTrain, aeScaler)       # fit and transform the training set
    xtest = transform(alphaTest, aeScaler)     # transform the test set

    if mode == 'podAE':

        # Shuffling data
        np.random.seed(cae_seed)
        perm = np.random.permutation(xtrain.shape[0])
        xtrain = xtrain[perm]

        # creating the AE model
        
        caeModel, encoder, decoder = createCAE1D(LrateCAE, AEmode,
                                            encoderDense=encoderDense, decoderDense=decoderDense,
                                            decoderReshape=decoderReshape, inputShapeCAE=inputShapeCAE,
                                            encoderZipped=encoderZipped, decoderZipped=decoderZipped)
        '''
        caeModel, encoder, decoder = createMLP(LrateCAE, AEmode,
                                            encoderDenseMLP=encoderDenseMLP,
                                            inputShapeCAE=PODmode)
        '''
        # Create a callback that saves the model's weights
        checkpoint_path = dirModel +f'/CAEbestWeights.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                    save_best_only=True, mode='min',
                                    save_weights_only=True, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                    patience=10, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystopping]


        # training the AE
        history = caeModel.fit( x=xtrain, 
                                y=xtrain, 
                                batch_size=batchSizeCAE, epochs=epochsCAE,
                                #verbose='1',
                                #validation_split=0.2)
                                callbacks=callbacks_list, validation_split=0.2)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = dirPlot + f'/PODcaeModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlot + f'/PODcaeModelMAE.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)

        # saving 3 models for encoder part, decoder part, and all the AE
        caeModel.save(dirModel + f'/PODAEModel.h5')
        encoder.save(dirModel + f'/PODAEencoderModel.h5')
        decoder.save(dirModel + f'/PODAEdecoderModel.h5')
        
    elif mode == 'podAEtest':

        # loading the trained AE
        caeModel = tf.keras.models.load_model(dirModel + f'/PODAEModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model(dirModel + f'/PODAEencoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5', custom_objects={'my_swish':my_swish})

        # predicting test set with trained AE model
        output = caeModel.predict(xtest)#[:, :, 0]
        
        # Inverse transform to find the real values
        output = inverseTransform(output, aeScaler)

        # evaluate the model
        scores = caeModel.evaluate(xtest, xtest, verbose=0)
        print("%s: %.2f%%" % (caeModel.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (caeModel.metrics_names[1], scores[1]*100))
        
        # saving the predicted values
        filename = dirResult + f"/PODAE"
        np.save(filename, output) 


        # Using training set to obtain scaling parameters.
        # This part should be rewritten in another way
        series1 = encoder.predict(xtrain)
        lstmScaler = StandardScaler()
        scaledSeries1 = scale(series1, lstmScaler)

        # encoding test set
        series = encoder.predict(xtest)
        # scaling test set
        scaledSeries = transform(series, lstmScaler)

        # plotting the evolution of modes in time
        fileName = dirPlot + f'/caeOutput.png'
        trainLabel='Training set'
        testLabel='Test set'
        validLabel='True data'
        figNum = 1
        aveTime = int (0.75 * (trainStartTime + trainEndTime))
        subplotMode(figNum, time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                    scaledSeries, scaledSeries, scaledSeries1[aveTime:],\
                    validLabel, testLabel, trainLabel, fileName, px, py)

    elif mode == 'lstm':

        # loading encoder, decoder, and autoencoder model
        caeModel = tf.keras.models.load_model(dirModel + f'/PODAEModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model(dirModel +f'/PODAEencoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5', custom_objects={'my_swish':my_swish})

        # encoding the training set
        series = encoder.predict(xtrain)
        
        # scaling the encoded values before feeding them to LSTM net.
        lstmScaler = StandardScaler()
        scaledSeries = scale(series, lstmScaler)

        # created suitable shape (window data) to feed LSTM net
        xtrainLSTM, ytrainLSTM = windowAdapDataSet(scaledSeries, windowSize, timeScale)

        #Shuffling data
        np.random.seed(lstm_seed)
        perm = np.random.permutation(ytrainLSTM.shape[0])
        xtrainLSTM = xtrainLSTM[perm,:,:]
        ytrainLSTM = ytrainLSTM[perm,:]

        # creating LSTM model
        lstmModel = createLSTM(LrateLSTM, AEmode, numLstmNeu)

        # training the model
        history = lstmModel.fit(xtrainLSTM, ytrainLSTM, epochs=epochsLSTM, batch_size=batchSizeLSTM, validation_split=validationLSTM)

        # saving the trained LSTM model
        lstmModel.save(dirModel + f'/lstmModel.h5')

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
        fileName = dirPlot + f'/lstmModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlot + f'/lstmModelMAE.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)

    elif mode == 'lstmTest':
        
        # loading encoder and decoder model
        encoder = tf.keras.models.load_model(dirModel + f'/PODAEencoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5', custom_objects={'my_swish':my_swish})

        # Using training set to obtain scaling parameters.
        # This part should be rewritten in another way
        series1 = encoder.predict(xtrain)
        lstmScaler = StandardScaler()
        scaledSeries1 = scale(series1, lstmScaler)

        # encoding test set
        series = encoder.predict(xtest)
        # scaling test set
        scaledSeries = transform(series, lstmScaler)
        
        # load trained LSTM model
        lstmModel = tf.keras.models.load_model(dirModel + f'/lstmModel.h5')
        
        # predicting future data using LSTM model for test set
        ytest = lstmTest(scaledSeries, lstmModel, windowSize, timeScale)

        err = np.linalg.norm(ytest - scaledSeries)/np.sqrt(np.size(ytest))
        print('err = ', err)
        
        # inverse transform of lstm prediction before decoding
        pred = inverseTransform(ytest, lstmScaler)
        # decoding the data
        output = decoder.predict(pred)#[:, :, 0]

        # inverse transform of AE prediction
        inverseOutput = inverseTransform(output, aeScaler)

        #Reconstruction
        PhidTrain = PhidTrain.T
        inverseOutput = inverseOutput.T
        dataRecons = np.dot(PhidTrain,inverseOutput)

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
        fileName = dirPlot + f'/lstmOutput.png'
        trainLabel='Training set'
        testLabel='Test set'
        validLabel='True data'
        figNum = 1
        aveTime = int (0.75 * (trainStartTime + trainEndTime))
        subplotMode(figNum, time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                    scaledSeries, ytest, scaledSeries1[aveTime:],\
                    validLabel, testLabel, trainLabel, fileName, px, py)

        predProbe = probe(reshapedData, lx, ly, mesh[0], mesh[1])
        trueProbe = probe(data, lx, ly, mesh[0], mesh[1])
        fileName = dirPlot + f'/PODprobe.png'
        plotTitle = f'evolution of {var}'

        subplotProbe(figNum, time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                    trueProbe[testStartTime:testEndTime], predProbe, trueProbe[aveTime:trainEndTime],\
                    validLabel, testLabel, trainLabel, fileName, px, py, var)

        #subplotProbe(figNum, time[testStartTime:testEndTime], trueProbe, predProbe, testLabel, validLabel,\
        #        '', '', plotTitle, fileName, mesh, lx, ly)

        # saving the AE-LSTM prediction
        filename = dirResult + f"/ROM"
        np.save(filename, inverseOutput)

    else:
        exit()
    
if __name__ == "__main__":
    main()
