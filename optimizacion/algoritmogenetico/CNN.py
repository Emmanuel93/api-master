import cPickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
from keras import callbacks
import csv
import SerieToImage
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import Helper as helper
import time
import matplotlib.pyplot as plt
import os


# fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)

def guardarExcel(archivoCSV, tar, out, nb_columna1, nb_columna2):

    with open(archivoCSV, "wb") as ArchivoNuevoCSV:

        writer = csv.writer(ArchivoNuevoCSV, lineterminator='\n')

        writer.writerow([str(nb_columna1), str(nb_columna2)])

        for i in range(0, len(tar)):
            tarTemp = str(tar[i])
            outTemp = str(out[i])
            aux1 = tarTemp.replace("[", "")
            tarAdd = aux1.replace("]", "")

            aux2 = outTemp.replace("[", "")
            outAdd = aux2.replace("]", "")

            writer.writerow( (tarAdd, outAdd) )

def fitModel(datosImagenesEntrenamiento, datosTargetEntrenamiento, tamanioImagen, epocas, valorDropout, optimizer, activation, convolutionalLayer1, convolutionalLayer2, poolingLayer1, poolingLayer2):

    # build the model
    model = baseline_model(tamanioImagen, valorDropout, optimizer, activation, convolutionalLayer1, convolutionalLayer2, poolingLayer1, poolingLayer2)

    #csv_logger = callbacks.CSVLogger('training.log', separator=',', append=True)

    # Fit the model
    #model.fit(datosImagenesEntrenamiento, datosTargetEntrenamiento, epochs=epocas, verbose=0, callbacks=[csv_logger])
    print "Empezo el entrenamiento"
    model.fit(datosImagenesEntrenamiento, datosTargetEntrenamiento, batch_size=10000 ,epochs=epocas, verbose=0)
    print "Finalizo el entrenamiento"
    
    return model

def baseline_model(tamanioImagen, valorDropout, optimizer, activation, convolutionalLayer1, convolutionalLayer2, poolingLayer1, poolingLayer2):
    # create model
    model = Sequential()
    model.add(convolutionalLayer1)
    model.add(activation)
    model.add(poolingLayer1)

    model.add(convolutionalLayer2)
    model.add(activation)
    model.add(poolingLayer2)

    # Dropout va de 0 - 1
    #model.add(Dropout(valorDropout))
    model.add(Flatten())

    model.add(Dense(128))
    #model.add(Dropout(valorDropout))
    model.add(Activation('relu'))

    #model.add(Dense(32))
    #model.add(Dropout(valorDropout))
    #model.add(Activation('relu'))

    model.add(Dropout(valorDropout))

    model.add(Dense(10))
    model.add(Activation('linear'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

def cross_validation():

    # Hora y fecha inicio
    horaInicio = time.strftime("%H-%M-%S")
    fechaInicio = time.strftime("%d-%m-%Y")

    nombreArchivo = "wind_el_fresno_serie_3000_tallerML"
    rutaArchivo = nombreArchivo+'.csv'
    serieOriginal = leerArchivo(rutaArchivo)
    serieOriginal = helper.normalizarSerie(serieOriginal)

    tamanioSerieTiempo = 8000

    serie = serieOriginal[:tamanioSerieTiempo]

    tamanioImagen = 24

    tipoSerieAImagen = 2

    if tipoSerieAImagen == 1:
        bancoImagenes = SerieToImage.obtenerBancoDeImagenes(tamanioImagen, tamanioImagen, serie)
    else:
        bancoImagenes = SerieToImage.obtenerBancoDeImagenes2(tamanioImagen, serie)

    datosImagenes, datosTarget = obtenerDataSet(bancoImagenes)

    datosImagenes = np.array(datosImagenes)
    datosTarget = np.array(datosTarget)

    data_train, data_test, target_train, target_test = train_test_split(datosImagenes, datosTarget, test_size=0.33, random_state=0)
    epocas = 1000

    kf = KFold(n_splits = 5)

    for valorDropout in np.arange(0.2, 0.8, 0.05):

        mseTotal = 0

        nCrossValidationFold = 0
        for X_train_index, X_test_index in kf.split(data_train):

            X_train = data_train[X_train_index]
            X_test = data_train[X_test_index]

            y_train = target_train[X_train_index]
            y_test = target_train[X_test_index]

            # reshape to be [samples][pixels][width][height]
            X_train = X_train.reshape(X_train.shape[0], 1, tamanioImagen, tamanioImagen).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], 1, tamanioImagen, tamanioImagen).astype('float32')

            # normalize inputs from 0-255 to 0-1
            X_train = X_train / 255
            X_test = X_test / 255

            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)

            model = fitModel(X_train, y_train, tamanioImagen, epocas, valorDropout)

            #nombreModelo = 'model_'+str(tamanioImagen)+'x'+str(tamanioImagen)+'_'+str(tamanioSerieTiempo)+'_epocas-'+str(epocas)+'.h5'

            #model.save(nombreModelo)

            #model = load_model(nombreModelo)

            predicted = model.predict(X_test)

            nombreResultado = 'resultados_'+fechaInicio+"_"+horaInicio+"_"+str(tamanioImagen)+"x"+str(tamanioImagen)+"_"+str(tamanioSerieTiempo)+'_epocas_'+str(epocas)+"_dropout_"+str(valorDropout)+"_fold_"+str(nCrossValidationFold)+'.csv'

            guardarExcel(nombreResultado, y_test, predicted, 'tar', 'out')

            mseTest = helper.MSE(y_test, predicted)

            mseTotal+= mseTest

            #mseTrain = MSE(y_train, predicted)

            nCrossValidationFold += 1

        nombreIteracion = str(tamanioImagen)+"x"+str(tamanioImagen)+"_"+str(tamanioSerieTiempo)+'_epocas_'+str(epocas)

        msePromedio = mseTotal / kf.get_n_splits(data_train)

        guardarMSEContinuoExcel("resultadosMSE"+fechaInicio+"_"+horaInicio+".csv", nombreIteracion, [mseTotal], 'MSE_TOTAL', [valorDropout], 'DROP_OUT')

        #print "MSETOTAL :", mseTotal, " _Dropout : ",valorDropout


    # Hora y fecha inicio
    horaFin = time.strftime("%H:%M:%S")
    fechaFin = time.strftime("%d/%m/%Y")

    print "Inicio: ", fechaInicio," ", horaInicio, " terminio: ",fechaFin, " ", horaFin

def rotar(datosImagenesTrain, datosImagenesTrainTarget, grados):

    nuevoDatosImagenesTrain = []
    nuevoDatosImagenesTrainTarget = []

    for datosImagenTrainIndex in range(len(datosImagenesTrain)):

        datosImagenTrain = datosImagenesTrain[datosImagenTrainIndex]

        datosImagenTrainRotada = SerieToImage.rotarImagen(datosImagenTrain, grados)

        nuevoDatosImagenesTrain.append(datosImagenTrain)
        nuevoDatosImagenesTrain.append(datosImagenTrainRotada)

        target = datosImagenesTrainTarget[datosImagenTrainIndex]

        nuevoDatosImagenesTrainTarget.append(target)
        nuevoDatosImagenesTrainTarget.append(target)

    return np.array(nuevoDatosImagenesTrain), np.array(nuevoDatosImagenesTrainTarget)

def simulacion():

    nombreArchivo = "sector19"
    rutaArchivo = nombreArchivo+'.csv'
    serieOriginal = leerArchivo(rutaArchivo)

    tamanioSerieTiempo = 8000

    serie = serieOriginal[:tamanioSerieTiempo]

    # Datos de entrenamiento = 80 % de la serie original
    data_train, data_test = train_test_split(serie, test_size=0.20, shuffle=False)

    # Mejor tamanio de imagen encontrado en experimiento de tamanios de imagenes
    tamanioImagen = 24
    numeroPaso = 1

    tipoSerieAImagen = 'GRAMIAN_ANGULAR_FIELD'

    # obtener el banco de imagenes para pruebas
    if tipoSerieAImagen == 'ESCANEO_LINEAL':
        bancoImagenesTest = SerieToImage.obtenerBancoDeImagenes(tamanioImagen, tamanioImagen, data_test)
    elif tipoSerieAImagen == 'GRAMIAN_ANGULAR_FIELD':
        bancoImagenesTest = SerieToImage.obtenerBancoDeImagenes2(tamanioImagen, data_test, numeroPaso)

    datosImagenesTest, datosImagenesTestTarget = obtenerDataSet(bancoImagenesTest)

    datosImagenesTest = np.array(datosImagenesTest)
    datosImagenesTestTarget = np.array(datosImagenesTestTarget)

    nombreModelo = 'Exprimiento4AguaSector19SinSuavizar_resultados_dataset07-12-2017_13-37-45_24x24_1080_epocas73_dropout0.4_tipoSerieAImagenGRAMIAN_ANGULAR_FIELD_paso:1.h5'

    model = load_model(nombreModelo)

    X_test = datosImagenesTest

    y_test = datosImagenesTestTarget

    # reshape to be [samples][pixels][width][height]
    X_test = X_test.reshape(X_test.shape[0], 1, tamanioImagen, tamanioImagen)

    # normalize inputs from 0-255 to 0-1
    X_test = X_test / 255

    y_test = y_test.reshape(len(y_test), 1)

    prediccionEnTest = model.predict(X_test)

    mseTest = helper.MSE(y_test, prediccionEnTest)

    print mseTest


def experimento(serie, epocas, learningRate, trainingRate, optimizer, activation, filterSize, strides, padding, pool, valorDropout, numeroPaso):

    # Hora y fecha inicio
    #horaInicio = time.strftime("%H-%M-%S")
    #fechaInicio = time.strftime("%d-%m-%Y")

    # Mejor tamanio de imagen encontrado en experimiento de tamanios de imagenes
    tamanioImagen = 28
    # Mejor dropout encontrado en experimiento de dropout, ahora este valor lo pasa el genetico
    #valorDropout = 0.4
    # the data, shuffled and split between train and test sets
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, tamanioImagen, tamanioImagen)
        x_test = x_test.reshape(x_test.shape[0], 1, tamanioImagen, tamanioImagen)
        input_shape = (1, tamanioImagen, tamanioImagen)
    else:
        x_train = x_train.reshape(x_train.shape[0], tamanioImagen, tamanioImagen, 1)
        x_test = x_test.reshape(x_test.shape[0], tamanioImagen, tamanioImagen, 1)
        input_shape = (tamanioImagen, tamanioImagen, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    opt = helper.optimizerFactory(optimizer, learningRate)
    act = helper.activationFactory(activation)

    type = '2d'
    filterNumber = 64
    kernelSize = (filterSize, filterSize)
    inputShape = input_shape
    padd = padding

    conv1 = helper.convolutionLayerFactory(type, filterNumber, kernelSize, inputShape, padd)
    conv2 = helper.convolutionLayerFactory(type, filterNumber, kernelSize, inputShape, padd)

    pool1 = helper.poolingFactory(pool)
    pool2 = helper.poolingFactory(pool)

    model = fitModel(x_train, y_train, tamanioImagen, epocas, valorDropout, opt, act, conv1, conv2, pool1, pool2)
    prediccionEnTest = model.predict(X_test)


    score = model.evaluate(x_test, y_test, verbose=0)
    #prediccionEnTrain = model.predict(X_train)

    mseTest = helper.MSE(y_test, prediccionEnTest)

    #mseTrain = helper.MSE(y_train, prediccionEnTrain)

    return score, mseTest, model

    '''
    if mseTest < mejorMSETest:

        mejorPrediccionTestNombre = idExperimento+'_resultados_'+"dataset"+fechaInicio+"_"+horaInicio+"_"+str(tamanioImagen)+"x"+str(tamanioImagen)+"_"+str(len(serie))+"_epocas"+str(epocas)+"_dropout"+str(valorDropout)+"_tipoSerieAImagen"+tipoSerieAImagen+"_paso:"+str(numeroPaso)

        mejorMSETest = mseTest

        mejorPrediccionTest = prediccionEnTest

        mejorModelo = model

    #nombreIteracion = idExperimento+"_"+fechaInicio+"_"+horaInicio+"_"+str(tamanioImagen)+"x"+str(tamanioImagen)+"_"+str(len(serie))+'_epocas:'+str(epocas)+"_dropout"+str(valorDropout)+"_tipoSerieAImagen"+tipoSerieAImagen+"_paso:"+str(numeroPaso)

    #guardarMSEContinuoExcel(idExperimento+"_resultadosMSE"+fechaInicio+"_"+horaInicio+".csv", nombreIteracion, [mseTest], 'MSE_TEST', [mseTrain], 'MSE_TRAIN')

    #nombreModelo = mejorPrediccionTestNombre+'.h5'

    #mejorModelo.save(nombreModelo)

    #guardarExcel(mejorPrediccionTestNombre+'.csv', y_test, mejorPrediccionTest, 'tar', 'out')

    #print "MSETOTAL :", mseTotal, " _Dropout : ",valorDropout

    # Hora y fecha inicio
    horaFin = time.strftime("%H:%M:%S")
    fechaFin = time.strftime("%d/%m/%Y")

    print "Inicio: ", fechaInicio," ", horaInicio, " terminio: ",fechaFin, " ", horaFin

    '''

def pronostico(serie, trainingRate, modelo, numeroPaso):

    # Mejor tamanio de imagen encontrado en experimiento de tamanios de imagenes
    tamanioImagen = 24
    tipoSerieAImagen = 'GRAMIAN_ANGULAR_FIELD'

    datosImagenesTrain, datosImagenesTrainTarget, datosImagenesTest, datosImagenesTestTarget = SerieToImage.obtenerImagenes(serie, tipoSerieAImagen, numeroPaso, tamanioImagen, trainingRate)

    X_train = datosImagenesTrain
    X_test = datosImagenesTest

    y_train = datosImagenesTrainTarget
    y_test = datosImagenesTestTarget

    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, tamanioImagen, tamanioImagen)
    X_test = X_test.reshape(X_test.shape[0], 1, tamanioImagen, tamanioImagen)

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    prediccionEnTest = modelo.predict(X_test)

    return y_test, prediccionEnTest

def graficaModelos():

    series = ['wind_elfresno_10m_complete_4673-9088']

    f, axarr = plt.subplots(2, sharex=True)
    f.suptitle('Sharing X axis')

    for nombreSerie in series:

        print nombreSerie

        for numeroPaso in range(18, 25, 6):

            #nombreArchivo = nombreSerie+"_normalizado"

            nombreArchivo = nombreSerie

            print nombreArchivo

            #nombreArchivo = nombreSerie

            rutaArchivo = nombreArchivo+'.csv'
            serie = helper.leerArchivo(rutaArchivo)

            serie = helper.normalizarSerie(serie)

            nombreModelo = nombreSerie+'_'+str(numeroPaso)+'.h5'

            #nombreModelo = 'mejorModeloPaso'+str(numeroPaso)+'.h5'

            model = load_model(nombreModelo)

            # No importa el valor del training rate ya que siempre aqui solo se usa
            # el validation set y esta fijo al 20% del total de la serie

            trainingRate = 1

            y_test, prediccionEnTest = pronostico(serie, trainingRate, model, numeroPaso)

            mseTest = helper.MSE(y_test, prediccionEnTest)

            print "y_test_paso:"+str(numeroPaso), y_test
            print "prediccionEnTest_paso:"+str(numeroPaso), prediccionEnTest


            #axarr[numeroPaso - 1 ].plot(y_test)
            #axarr[numeroPaso - 1 ].plot(prediccionEnTest)

            numeroPaso = numeroPaso + 1

    plt.show()


#1.- Esta funcion suviza el TS de una serie utilizando Running Meddian (Metodo Glenn)
#2.- Normaliza TS tomando su maximo y minimo y el VS tomando el maximo y minimo del TS
def genearSeriesHector():

    rutaSeries = ''
    series = ['wind_el_fresno_serie_sin_huecos']
    trainiSetPorcent = 0.80
    validationSetPorcent = 1 - trainiSetPorcent

    for nombreSerie in series:

        rutaArchivo = rutaSeries+nombreSerie+'.csv'
        serie = helper.leerArchivo(rutaArchivo)

        validationSetSize = int(len(serie) * validationSetPorcent)

        data_train, data_test = train_test_split(serie, test_size=validationSetSize, shuffle=False)

        data_train = helper.running_median_insort(data_train, 3)

        maxTrainingSet = max(data_train)
        minTrainingSet = min(data_train)

        data_train = helper.normalizarSerieConMaximoMinimo(data_train, maxTrainingSet, minTrainingSet)

        data_test = helper.normalizarSerieConMaximoMinimo(data_test, maxTrainingSet, minTrainingSet)

        data_train = np.array(data_train).reshape(len(data_train), 1)

        data_test = np.array(data_test).reshape(len(data_test), 1)

          # obtener el banco de imagenes para entrenamiento

        # GRAMIAN_ANGULAR_FIELD
        tamanioImagen = 24
        numeroPaso = 1

        bancoImagenesTrain = SerieToImage.obtenerBancoDeImagenes2(tamanioImagen, data_train, numeroPaso)

        datosImagenesTrain, datosImagenesTrainTarget = SerieToImage.obtenerDataSet(bancoImagenesTrain)

        datosImagenesTrain = np.array(datosImagenesTrain)
        datosImagenesTrainTarget = np.array(datosImagenesTrainTarget)

        #GRAMIAN_ANGULAR_FIELD
        bancoImagenesTest = SerieToImage.obtenerBancoDeImagenes2(tamanioImagen, data_test, numeroPaso)

        datosImagenesTest, datosImagenesTestTarget = SerieToImage.obtenerDataSet(bancoImagenesTest)

        datosImagenesTest = np.array(datosImagenesTest)
        datosImagenesTestTarget = np.array(datosImagenesTestTarget)

        np.savetxt(rutaSeries+nombreSerie+'TS.csv', data_train)
        np.savetxt(rutaSeries+nombreSerie+'VS.csv', data_test)

        '''
        path = "SeriesHector/imagenes/"+nombreSerie
        os.mkdir(path)

        pathTraining = path +"/training/"
        os.mkdir(pathTraining)

        pathValidation = path + "/validation/"
        os.mkdir(pathValidation)

        #SerieToImage.datosAImagen(datosImagenesTrain, datosImagenesTrainTarget, pathTraining)

        #SerieToImage.datosAImagen(datosImagenesTest, datosImagenesTestTarget, pathValidation)
        '''


if __name__ == "__main__":

    #genearSeriesHector()
    graficaModelos()

    '''
    import cProfile,pstats
    cProfile.run('experimento()','.prof')
    prof = pstats.Stats('.prof')
    prof.strip_dirs().sort_stats('time').print_stats(30)
    '''

    #experimento()
