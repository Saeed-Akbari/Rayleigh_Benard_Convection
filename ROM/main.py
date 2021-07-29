""" @author: Saeed  """

from os import times
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


from preprocess import loadData, loadMesh, splitFlattenData, splitData,\
                        scale, transform, inverseTransform,\
                        windowed_dataset, createLSTMdata
from visualization import animationGif, contourPlot, contourSubPlot, plot, plot1, subplot
from CAE import createCAE
from LSTM import createLSTM
from activation import my_swish

from derTest import f2th, fx4th, fy4th

def main():
    
    loc = '../FOM/'
    dataPsi, dataTh, flattenedPsi, flattenedTh, time, mesh = loadData(loc)
    Xmesh, Ymesh = loadMesh(loc)
    #animationGif(Xmesh, Ymesh, dataPsi, fileName='Psi', figSize=(14,7))
    #contourPlot(Xmesh, Ymesh, dataPsi[:, :, 80], fileName='CFD', figSize=(14,7))
    '''
    x = np.arange(10)
    y = np.arange(10)
    Ymesh, Xmesh = np.meshgrid(x, y)
    dataPsi = np.empty((2, x.shape[0], y.shape[0]))
    dataPsi[0, :, :] = Xmesh*Xmesh
    dataPsi[1, :, :] = Ymesh*Ymesh
    '''
    '''
    u2, v2 = f2th(dataPsi, Xmesh, Ymesh)
    
    u4 = fx4th(dataPsi, Xmesh)
    v4 = fy4th(dataPsi, Ymesh)
    
    print("x component of vecocity second order: ", u2)
    print("x component of vecocity fourth order: ", u4)
    print("y component of vecocity second order: ", v2)
    print("y component of vecocity fourth order: ", v4)
    print("x component of vecocity: ", np.linalg.norm(u4-u2)/np.linalg.norm(u2))
    print()
    print("y component of vecocity: ", np.linalg.norm(v4-v2)/np.linalg.norm(u2))
    '''
    '''
    with open('inputParam.yaml') as inputParamFile:    
        input_data = yaml.load(inputParamFile, Loader=yaml.FullLoader)
    inputParamFile.close()
    '''
    
    trainStartTime = 50
    trainEndTime = 80
    testStartTime = 70
    testEndTime = 95
    #   time[np.argwhere(time>trainStartTime)][:, 0][0, 0].round(1)
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0]
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0]
    testStartTime = np.argwhere(time>testStartTime)[0, 0]
    testEndTime = np.argwhere(time<testEndTime)[-1, 0]
    trainDataLen = trainEndTime - trainStartTime
    testDataLen = testEndTime - testStartTime
    
    figTimeTrain = np.array([55, 65, 75])
    figTimeTest = np.array([85, 90])
    for i in range(figTimeTrain.shape[0]):
        figTimeTrain[i] = np.argwhere(time>figTimeTrain[i])[0, 0]
    figTimeTrain = figTimeTrain - trainStartTime
    for i in range(figTimeTest.shape[0]):
        figTimeTest[i] = np.argwhere(time>figTimeTest[i])[0, 0]
    figTimeTest = figTimeTest - testStartTime

    epochsCAE = 2000
    batchSizeCAE = 32
    LrateCAE = 1e-5
    latentLayer = 9
    encoderLayer = 4
    numChannel = 1
    inputShapeCAE = ((mesh[0]+1), (mesh[1]+1), numChannel)
    encoderFilters = [16, 8]
    encoderDense = [1024, 128]
    decoderDense = [128, 512]
    decoderReshape = (8, 8, 8)
    decoderFilters = [16, 32, 64]
    filtersKey = ["encoderFilters", "encoderDense", "decoderFilters", "decoderDense", "decoderReshape", "inputShapeCAE"]
    filtersList = [encoderFilters, encoderDense, decoderFilters, decoderDense, decoderReshape, inputShapeCAE]
    filters = {}
    for key in filtersKey:
        for value in filtersList:
            filters[key] = value
            filtersList.remove(value)
            break

    #mode = input("Enter either \"cae\" for convolutional autoencoder or \"lstm\" for " +
    #            "large short term memory:")
    
    mode = 'lstmTest'    # cae, lstm, load_models, lstmTest, caeTest
    
    windowSize = 5
    batchSizeLSTM = 32
    LrateLSTM = 1e-3
    epochsLSTM = 150
    shuffleBufferSize = 1000

    dataPsiTrain = splitData(dataPsi, dataPsi.shape[0], trainStartTime, trainEndTime)
    dataPsiTest = splitData(dataPsi, dataPsi.shape[0], testStartTime, testEndTime)
    dataThTrain = splitData(dataTh, dataTh.shape[0], trainStartTime, trainEndTime)
    dataThTest = splitData(dataTh, dataTh.shape[0], testStartTime, testEndTime)
    flattenedPsiTrain = splitFlattenData(flattenedPsi, flattenedPsi.shape[0], trainStartTime, trainEndTime)
    flattenedPsiTest = splitFlattenData(flattenedPsi, flattenedPsi.shape[0], testStartTime, testEndTime)
    flattenedThTrain = splitFlattenData(flattenedTh, flattenedTh.shape[0], trainStartTime, trainEndTime)
    flattenedThTest = splitFlattenData(flattenedTh, flattenedTh.shape[0], testStartTime, testEndTime)

    # Scale the training data
    PsiScaler = StandardScaler()
    scaledPsiTrain = scale(flattenedPsiTrain, PsiScaler)            # temporary     flattenedPsiTrain, flattenedThTrain
    scaledPsiTest = transform(flattenedPsiTest, PsiScaler)
    ThScaler = StandardScaler()
    #scaledThTrain = scale(flattenedThTrain, ThScaler)
    #scaledThTest = transform(flattenedThTest, ThScaler)

    '''
    #       Randomize the data later

    # Randomize training data
    #idx =  np.arange(PsiTrain.shape[1])
    #np.random.shuffle(idx)
    #sPsiTrainRandomized = PsiTrain[idx[:]]
    '''
    PsiTrainRandomized = scaledPsiTrain
    #ThTrainRandomized = scaledThTrain

    reversedFlattenPsi = PsiTrainRandomized.reshape(trainDataLen, (mesh[0]+1),
                                                    (mesh[1]+1))
    reversedFlattenTest = scaledPsiTest.reshape(testDataLen, (mesh[0]+1),
                                                    (mesh[1]+1))
    
    num_example = np.shape(reversedFlattenPsi)[0]
    processdPsiTrain = np.empty((num_example, mesh[0]+1, mesh[1]+1, 1)) # Channels last
    for i in range(num_example):
        temp = reversedFlattenPsi[i, :, :].reshape((mesh[0]+1), (mesh[1]+1))
        processdPsiTrain[i,:,:,0] = temp[:,:]

    num_test = np.shape(reversedFlattenTest)[0]
    processdPsiTest = np.empty((num_test, mesh[0]+1, mesh[1]+1, 1)) # Channels last
    for i in range(num_test):
        temp = reversedFlattenTest[i, :, :].reshape((mesh[0]+1), (mesh[1]+1))
        processdPsiTest[i,:,:,0] = temp[:,:]
    
    if mode == 'cae':
        caeModel, encoder, decoder = createCAE(LrateCAE, latentLayer,
                                            encoderFilters=encoderFilters, encoderDense=encoderDense,
                                            decoderFilters=decoderFilters, decoderDense=decoderDense,
                                            decoderReshape=decoderReshape, inputShapeCAE=inputShapeCAE)
        
        # Create a callback that saves the model's weights
        checkpoint_path = './model/CAEbestWeights.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                    save_best_only=True, mode='min',
                                    save_weights_only=True, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                    patience=10, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystopping]

        history = caeModel.fit(x=processdPsiTrain, 
                                y=processdPsiTrain, 
                                batch_size=batchSizeCAE, epochs=epochsCAE,
                                verbose='2',
                                #validation_split=0.1)
                                callbacks=callbacks_list, validation_split=0.1)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = './plot/caeModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = './plot/caeModelMAE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        output = caeModel.predict(processdPsiTrain)[:, :, :, 0]
        
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()
        flattenedOutput = inverseTransform(flattenedOutput, PsiScaler)

        inverseOutput = flattenedOutput.reshape((output.shape[0],  output.shape[1], output.shape[2]))

        caeModel.save('./model/caeModel.h5')
        encoder.save('./model/encoderModel.h5')
        decoder.save('./model/decoderModel.h5')


    elif mode == 'caeTest':
        caeModel = tf.keras.models.load_model('./model/caeModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model('./model/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model('./model/decoderModel.h5', custom_objects={'my_swish':my_swish})

        output = caeModel.predict(processdPsiTest)
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()
        flattenedOutput = inverseTransform(flattenedOutput, PsiScaler)
        inverseOutput = flattenedOutput.reshape((output.shape[0],  output.shape[1], output.shape[2]))
        contourSubPlot(Xmesh, Ymesh, dataPsiTest[figTimeTest[0], :, :], inverseOutput[figTimeTest[0], :, :],
                    dataPsiTest[figTimeTest[1], :, :], inverseOutput[figTimeTest[1], :, :], fileName='./plot/CAE', figSize=(14,7))

        
    elif mode == 'lstm':

        caeModel = tf.keras.models.load_model('./model/caeModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model('./model/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model('./model/decoderModel.h5', custom_objects={'my_swish':my_swish})

        series = encoder.predict(processdPsiTrain)
        lstmScaler = StandardScaler()
        scaledSeries = scale(series, lstmScaler)
        #datasetPsiTrain = windowed_dataset(series.flatten(), windowSize, batchSizeLSTM,
        #                                shuffleBufferSize)
        
        xtrain, ytrain = createLSTMdata(scaledSeries, windowSize)

        '''
        # Train LSTM
        encoded_list = []
        encoded_list.append(K.eval(encoder(processdPsiTrain.astype('float32'))))
        encoded = np.asarray(encoded_list)

        # Prepare training data
        lstmSeries = np.copy(encoded)[0, :, :]
        num_train_snapshots = 1
        total_size = np.shape(lstmSeries)[0]
        
        # Shape the inputs and outputs  
        input_seq = np.zeros(shape=(total_size,windowSize,latentLayer))         #-windowSize*num_train_snapshots
        output_seq = np.zeros(shape=(total_size,latentLayer))

        # Setting up inputs
        sample = 0
        for t in range(windowSize,lstmSeries.shape[0]):
            input_seq[sample,:,:latentLayer] = lstmSeries[t-windowSize:t,:]
            output_seq[sample,:] = lstmSeries[t,:]
            sample = sample + 1
        print(input_seq)
        '''
        lstmModel = createLSTM(LrateLSTM, latentLayer)
        
        history = lstmModel.fit(xtrain, ytrain, epochs=epochsLSTM, batch_size=batchSizeLSTM, validation_split=0.1)

        output1 = lstmModel.predict(xtrain)
        inverseOutput1 = inverseTransform(output1, lstmScaler)
        output = decoder.predict(inverseOutput1)[:, :, :, 0]

        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()

        inverseOutput = inverseTransform(flattenedOutput, PsiScaler)
        inverseOutput = inverseOutput.reshape(xtrain.shape[0], (mesh[0]+1), (mesh[1]+1))

        lstmModel.save('./model/lstmModel.h5')

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
        fileName = './plot/lstmModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = './plot/lstmModelMAE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

    elif mode == 'lstmTest':
        
        encoder = tf.keras.models.load_model('./model/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model('./model/decoderModel.h5', custom_objects={'my_swish':my_swish})
        lstmScaler = StandardScaler()
        series = encoder.predict(processdPsiTest)
        scaledSeries = scale(series, lstmScaler)
        
        xtest, ytest = createLSTMdata(scaledSeries, windowSize)
        lstmModel = tf.keras.models.load_model('./model/lstmModel.h5')

        output1 = lstmModel.predict(xtest)
        inverseOutput1 = inverseTransform(output1, lstmScaler)
        output = decoder.predict(inverseOutput1)[:, :, :, 0]

        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()

        inverseOutput = inverseTransform(flattenedOutput, PsiScaler)
        inverseOutput = inverseOutput.reshape(xtest.shape[0], (mesh[0]+1), (mesh[1]+1))
        contourSubPlot(Xmesh, Ymesh, dataPsiTest[figTimeTest[0], :, :], inverseOutput[figTimeTest[0], :, :],
                    dataPsiTest[figTimeTest[1], :, :], inverseOutput[figTimeTest[1], :, :], fileName='./plot/LSTM', figSize=(14,7))

        '''
        print(scaledSeries[windowSize:windowSize+windowSize, 0])
        print(ytest[:windowSize, 0])
        print(scaledSeries[-1-windowSize:, 0])
        print(ytest[-1-windowSize:, 0])
        '''

        '''
        print(scaledSeries[:windowSize, 0])
        print(xtest[0, :, 0])
        #print(scaledSeries[-1-windowSize:-1, 0])
        #print(xtest[-1-windowSize:, :, 0])
        #print(xtest.shape)
        '''

        '''
        for i in range(latentLayer):
            figNum = i + 2
            fileName = f'./plot/lstmOutput{i+1}.png'
            plotTitle = f'Latent Dimension{i+1}'
            trainLabel='Prediction'
            validLabel='True'
            xlabel = "Temporal snapshot"
            ylabel = "Magnitude"
            plot1(figNum, range(ytest.shape[0]), output1[:, i], ytest[:, i], trainLabel, validLabel, xlabel, ylabel, plotTitle, fileName)
        '''
        
        fileName = f'./plot/lstmOutput.png'
        plotTitle = f'Latent Dimension'
        trainLabel='Prediction'
        validLabel='True'
        figNum = 1
        subplot(figNum, range(ytest.shape[0]), output1, ytest, trainLabel, validLabel, "Temporal snapshot", "Magnitude", plotTitle, fileName)

        print("lstmTest")

    elif mode == 'load_models':
    
        caeModel = tf.keras.models.load_model('./model/caeModel.h5', custom_objects={'my_swish':my_swish})
        encoder = tf.keras.models.load_model('./model/encoderModel.h5', custom_objects={'my_swish':my_swish})
        decoder = tf.keras.models.load_model('./model/decoderModel.h5', custom_objects={'my_swish':my_swish})
        lstmModel = tf.keras.models.load_model('./model/lstmModel.h5')

        output = caeModel.predict(processdPsiTrain)[:, :, :, 0]
        print("processdPsiTrain shape = ", processdPsiTrain.shape)
        print("output shape = ", output.shape)
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()
        flattenedOutput = inverseTransform(flattenedOutput, PsiScaler)
        inverseOutput = flattenedOutput.reshape((output.shape[0],  output.shape[1], output.shape[2]))

        series = encoder.predict(processdPsiTrain)
        lstmScaler = StandardScaler()
        scaledSeries = scale(series, lstmScaler)
        xtrain, ytrain = createLSTMdata(scaledSeries, windowSize)
        output = lstmModel.predict(xtrain)
        inverseOutput = inverseTransform(output, lstmScaler)
        output = decoder.predict(inverseOutput)[:, :, :, 0]
        flattenedOutput = np.empty((output.shape[0], output.shape[1] * output.shape[2]))
        for i in range(output.shape[0]):
            flattenedOutput[i, :] = output[i, :, :].flatten()
        inverseOutput = inverseTransform(flattenedOutput, PsiScaler)
        inverseOutput = inverseOutput.reshape(xtrain.shape[0], (mesh[0]+1), (mesh[1]+1))

        # evaluate the model
        scores = caeModel.evaluate(processdPsiTrain, processdPsiTrain, verbose=0)
        print("%s: %.2f%%" % (caeModel.metrics_names[1], scores[1]*100))
        scores = lstmModel.evaluate(xtrain, ytrain, verbose=0)
        print("%s: %.2f%%" % (lstmModel.metrics_names[1], scores[1]*100))
        print("output data = ", inverseOutput.shape)
        print("input data = ", processdPsiTrain.shape)

        '''
        testing_set = np.copy(atrue)
        testing_set_scaled = sc.fit_transform(testing_set)
        testing_set= testing_set_scaled


        m,n = testing_set.shape
        ytest = np.zeros((1,lookback,n))
        ytest_ml = np.zeros((m,n))

        # create input at t = 0 for the model testing
        for i in range(lookback):
            ytest[0,i,:] = testing_set[i]
            ytest_ml[i] = testing_set[i]

        '''
        
        '''
        print("processdPsiTrain shape = ", processdPsiTrain[30, 5:15, 5, 0])
        print("output shape = ", output[30, 5:15, 5, 0])
        print("mean_squared_error = ", mean_squared_error(processdPsiTrain[30, :, :, 0], output[30, :, :, 0]))
        '''
    else:
        '''
        print("Mode should be changed to either CAE or LSTM")
        CAEdata = np.load('CAEdata.npy')
        CFDdata = np.load('CFDdata.npy')
        #print(np.linalg.norm(CAEdata[figTimeTrain[0], :, :]-CFDdata[:, :, figTimeTrain[0]]))
        print(CAEdata[figTimeTrain[0], :, :])
        print(CFDdata[:, :, figTimeTrain[0]])
        '''
        exit()
    
if __name__ == "__main__":
    main()