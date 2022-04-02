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
                        plot1, subplotMode, subplotProbe

from CAEmodel import createCAE
from LSTMmodel import createLSTM
from activation import my_swish

def main():
    
    with open('input.yaml') as file:    
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    mode = input_data['mode']
    var = input_data['var']
    
    epochsCAE = input_data['AE']['epochsCAE']
    batchSizeCAE = input_data['AE']['batchSizeCAE']
    LrateCAE = float(input_data['AE']['LrateCAE'])
    AEmode = input_data['AE']['AEmode']
    px = input_data['AE']['px']
    py = input_data['AE']['py']
    numChannel = input_data['AE']['numChannel']
    sn = input_data['AE']['sn']
    cae_seed = input_data['AE']['cae_seed']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']
    figTimeTest = np.array(input_data['figTimeTest'])
    epochsLSTM = input_data['AE']['epochsLSTM']
    batchSizeLSTM = input_data['AE']['batchSizeLSTM']
    LrateLSTM = float(input_data['AE']['LrateLSTM'])
    validationLSTM = input_data['AE']['validationLSTM']
    windowSize = input_data['AE']['windowSize']
    timeScale = input_data['AE']['timeScale']
    numLstmNeu = input_data['AE']['numLstmNeu']
    lstm_seed = input_data['AE']['lstm_seed']
    lx = input_data['AE']['lx']
    ly = input_data['AE']['ly']
    
    encoderFilters = input_data['AE']['encoderFilters']
    encoderPoolSize = input_data['AE']['encoderPoolSize']
    encoderStride = input_data['AE']['encoderStride']
    encoderDense = input_data['AE']['encoderDense']
    decoderDense = input_data['AE']['decoderDense']
    decoderReshape = input_data['AE']['decoderReshape']
    decoderFilters = input_data['AE']['decoderFilters']
    decoderPoolSize = input_data['AE']['decoderPoolSize']
    decoderStride = input_data['AE']['decoderStride']

    encoderZipped = zip(encoderFilters, encoderPoolSize, encoderStride)
    decoderZipped = zip(decoderFilters, decoderPoolSize, decoderStride)

    # loading data obtained from full-order simulation
    #The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
    loc = '../FOM/'
    #flattenedPsi, flattenedTh, time, dt, mesh = loadData(loc, var)
    flattened, dt, ra, mesh = loadData(loc, var)
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime, 0.1)

    # Convolutional autoencoder input shape
    inputShapeCAE = ((mesh[0]+1), (mesh[1]+1), numChannel)

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
    dirPlot = f'plot/{var}_AE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)
    # Creating a directory for models.
    dirModel = f'model/{var}_AE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirModel):
        os.makedirs(dirModel)
    # Creating a directory for result data.
    dirResult = f'result/{var}_AE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
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

    # splitting the data training and test set
    dataTrain = splitData(data, data.shape[0], trainStartTime, trainEndTime)
    dataTest = splitData(data, data.shape[0], testStartTime, testEndTime)
    flattenedTrain = splitData(flattened, flattened.shape[0], trainStartTime, trainEndTime)
    flattenedTest = splitData(flattened, flattened.shape[0], testStartTime, testEndTime)

    # Scale the training data
    scal = StandardScaler()
    scaledTrain = scale(flattenedTrain, scal)       # fit and transform the training set
    scaledTest = transform(flattenedTest, scal)     # transform the test set

    # reshape the sets to have their original shapes before feeding them into 2D convolutional AE
    reversedFlattenTrain = scaledTrain.reshape(trainDataLen, (mesh[0]+1),
                                                    (mesh[1]+1))
    reversedFlattenTest = scaledTest.reshape(testDataLen, (mesh[0]+1),
                                                    (mesh[1]+1))
    
    # creating training set in a suitable shape for AE net
    xtrain = np.empty((trainDataLen, mesh[0]+1, mesh[1]+1, numChannel))
    for i in range(trainDataLen):
        temp = reversedFlattenTrain[i].reshape((mesh[0]+1), (mesh[1]+1))
        xtrain[i,:,:,0] = temp

    # creating test set in a suitable shape for AE net
    xtest = np.empty((testDataLen, mesh[0]+1, mesh[1]+1, numChannel))
    for i in range(testDataLen):
        temp = reversedFlattenTest[i].reshape((mesh[0]+1), (mesh[1]+1))
        xtest[i,:,:,0] = temp

    if mode == 'cae':

        # skiping some snap shots
        #xtrain = xtrain[::sn]

        # Shuffling data
        np.random.seed(cae_seed)
        perm = np.random.permutation(xtrain.shape[0])
        xtrain = xtrain[perm]

        # creating the AE model
        caeModel, encoder, decoder = createCAE(LrateCAE, AEmode,
                                            encoderDense=encoderDense, decoderDense=decoderDense,
                                            decoderReshape=decoderReshape, inputShapeCAE=inputShapeCAE,
                                            encoderZipped=encoderZipped, decoderZipped=decoderZipped)
        
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
                                verbose='2',
                                #validation_split=0.1)
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
        fileName = dirPlot + f'/caeModelMSE_{mesh[0]+1}_{mesh[1]+1}_{dt}.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlot + f'/caeModelMAE_{mesh[0]+1}_{mesh[1]+1}_{dt}.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)

        # saving 3 models for encoder part, decoder part, and all the AE
        caeModel.save(dirModel + f'/caeModel.h5')
        encoder.save(dirModel + f'/encoderModel.h5')
        decoder.save(dirModel + f'/decoderModel.h5')

    elif mode == 'caeTest':
        
        # loading the trained AE
        caeModel = tf.keras.models.load_model(dirModel + f'/caeModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model(dirModel + f'/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/decoderModel.h5', custom_objects={'my_swish':my_swish})

        # predicting test set with trained AE model
        output = caeModel.predict(xtest)

        # Flattening the predicted values to have a suitable shape for transforming it the its original values
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()
        
        # Inverse transform to find the real values
        flattenedOutput = inverseTransform(flattenedOutput, scal)

        # reshaping again to have the original shape of the data
        inverseOutput = flattenedOutput.reshape((output.shape[0],  output.shape[1], output.shape[2]))

        # evaluate the model
        scores = caeModel.evaluate(xtest, xtest, verbose=0)
        #print("%s: %.2f%%" % (caeModel.metrics_names[1], scores[1]*100))
        #print("%s: %.2f%%" % (caeModel.metrics_names[0], scores[0]*100))

        # plotting the contour for the selected time to compare AE prediction with full-order model simulation
        contourSubPlot(Xmesh, Ymesh, dataTest[figTimeTest[0], :, :], inverseOutput[figTimeTest[0], :, :],
                    dataTest[figTimeTest[1], :, :], inverseOutput[figTimeTest[1], :, :], barRange,\
                    fileName=dirPlot + f'/CAE.png', figSize=(14,7))

        # saving the predicted values
        filename = dirResult + f"/CAE"
        np.save(filename, inverseOutput)


        # Using training set to obtain scaling parameters.
        # This part should be rewritten in another way
        series1 = encoder.predict(xtrain)
        lstmScaler = StandardScaler()
        scaledSeries = scale(series1, lstmScaler)

        # encoding test set
        series = encoder.predict(xtest)
        # scaling test set
        scaledSeries = transform(series, lstmScaler)

        # plotting the evolution of modes in time
        fileName = dirPlot + f'/caeOutput.png'
        plotTitle = ''
        testLabel='Prediction'
        validLabel='True'
        figNum = 1
        subplot(figNum, time[testStartTime:testEndTime], scaledSeries, scaledSeries, testLabel, validLabel,\
                "Temporal snapshot", "Magnitude", plotTitle, fileName, px, py)


    elif mode == 'lstm':

        # loading encoder, decoder, and autoencoder model
        caeModel = tf.keras.models.load_model(dirModel + f'/caeModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model(dirModel +f'/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/decoderModel.h5', custom_objects={'my_swish':my_swish})

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
        encoder = tf.keras.models.load_model(dirModel + f'/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model(dirModel + f'/decoderModel.h5', custom_objects={'my_swish':my_swish})
        
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
        output = decoder.predict(pred)[:, :, :, 0]
        
        # flattening the decoded data before inverse transformation of AE
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()

        # inverse transform of AE prediction
        inverseOutput = inverseTransform(flattenedOutput, scal)
        # reshape the data to its original shape
        inverseOutput = inverseOutput.reshape(output.shape[0], (mesh[0]+1), (mesh[1]+1))
        # plot the countor for AE-LSTM prediction for selected time
        contourSubPlot(Xmesh, Ymesh, dataTest[figTimeTest[0], :, :], inverseOutput[figTimeTest[0], :, :],
                    dataTest[figTimeTest[1], :, :], inverseOutput[figTimeTest[1], :, :], barRange,\
                    fileName= dirPlot + f'/LSTM.png', figSize=(14,7))
        
         
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

        predProbe = probe(inverseOutput, lx, ly, mesh[0], mesh[1])
        trueProbe = probe(data[testStartTime:testEndTime, :, :], lx, ly, mesh[0], mesh[1])
        fileName = dirPlot + f'/probe.png'
        plotTitle = f'evolution of {var}'
        subplotProbe(figNum, time[testStartTime:testEndTime], predProbe, trueProbe, testLabel, validLabel,\
                '', '', plotTitle, fileName, mesh, lx, ly)

        # saving the AE-LSTM prediction
        filename = dirResult + f"/ROM"
        np.save(filename, inverseOutput)
        
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
