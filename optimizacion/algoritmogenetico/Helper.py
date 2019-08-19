import csv
from collections import deque
from bisect import insort, bisect_left
from itertools import islice
import logging
from keras import optimizers
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D


def leerArchivo(ruta):
    serie = []

    with open(ruta, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            dato = float(row[0])
            serie.append(dato)
    return serie

def normalizarSerie(serie):
    minimo = float(min(serie))
    maximo = float(max(serie))
    dividendo = float(maximo-minimo)
    print dividendo
    return map(lambda x : (x - minimo) / dividendo, serie )

def normalizarSerieConMaximoMinimo(serie, maximo, minimo):

    return map(lambda x : (x - minimo) / (maximo - minimo), serie )    

def normalizarSerieParaImagen2(serie):
        minimo = float(min(serie))
        maximo = float(max(serie))

        return map(lambda x : ((x - maximo) +  (x - minimo)) / (maximo - minimo), serie )

def normalizarSerieParaImagen(serie):
    minimo = min(serie)
    maximo = max(serie)

    return map(lambda x : (x - minimo) * 255 / (maximo - minimo), serie )

def suavizarSeriePromedioMovible(serie):

    serieSuavizada = []

    MovingAverageSize = 3
    MovingAverageSizeAux = MovingAverageSize

    for index in range(len(serie)):

        datos = serie[index:MovingAverageSizeAux]

        if len(datos) < MovingAverageSize:
            break

        serieSuavizada.append(sum(datos) / MovingAverageSize )

        MovingAverageSizeAux += 1

    return serieSuavizada

def running_median_insort(seq, window_size):
	"""Contributed by Peter Otten"""
	seq = iter(seq)
	d = deque()
	s = []
	result = []
	for item in islice(seq, window_size):
		d.append(item)
		insort(s, item)
		result.append(s[len(d)//2])
	m = window_size // 2

	for item in seq:
		old = d.popleft()
		d.append(item)
		del s[bisect_left(s, old)]
		insort(s, item)
		result.append(s[m])
        
	return result

def MSE(originalList, forecastingList):
    sumatoria = 0
    for index, elemet in enumerate(originalList):
        forecasting = forecastingList[index]
        sumatoria = sumatoria + ((forecasting - elemet)**2)

    return sumatoria[0] / len(originalList)

def optimizerFactory(optimizer, learningRate):

    if optimizer == 'SGD':
        return optimizers.SGD(lr=learningRate)
    elif optimizer == 'ADAM':
        return optimizers.Adam(lr=learningRate)
    elif optimizer == 'RMSprop':
        return optimizers.RMSprop(lr=learningRate)
    else:
        logging.warning('Optimizer incorrecto !!')

        raise

def activationFactory(activation):

    if activation == 'relu':
        return Activation('relu')
    elif activation == 'elu':
        return Activation('elu')
    else :

        logging.warning('Activation incorrecto !!')

        raise

def convolutionLayerFactory(type, filterNumber, kernelSize, inputShape, padding):
    if type == '2d':
        return Conv2D(filterNumber, kernelSize, input_shape=(1,28,28), padding=padding,data_format='channels_first')
    else:
        logging.warning('Tipo Capa Convolucion incorrecto !!')

        raise

def poolingFactory(type):

    if type == 'MaxPooling2D':
        return MaxPooling2D(pool_size=(2, 2))
    elif type == 'AveragePooling2D':
        return AveragePooling2D(pool_size=(2,2))
    else :
        logging.warning('Tipo Pooling incorrecto !!')

        raise

def binarioADecimal(binario):

    binarioStr = "".join(str(x) for x in binario)

    return int(binarioStr, 2)

def guardarMSEContinuoExcel(dirArchivoCSV, tag1, valor1, tagValor1, valor2, tagValor2):

    with open(dirArchivoCSV, "a") as ArchivoNuevoCSV:

        writer = csv.DictWriter(ArchivoNuevoCSV, fieldnames= ['iteracion', tagValor1, tagValor2])

        for i in range(0, len(valor1)):
            valor1 = str(valor1[i])

            valor1 = valor1.replace("[", "")
            valor1 = valor1.replace("]", "")

            writer.writerow({'iteracion': tag1, tagValor1: valor1, tagValor2: valor2})

def formato(serie, tamanioVectorCaracteristicas):
    inp = []
    tar = []

    for i in range (0, len(serie) - tamanioVectorCaracteristicas ):

        input = serie[ i: i + tamanioVectorCaracteristicas]
        target = serie[ i + tamanioVectorCaracteristicas]

        inp.append(input)
        tar.append(target)

    return inp, tar


def formatoPaso (serie, tamanioVectorCaracteristicas, paso):

    inp = []
    tar = []

    for i in range (0, (len(serie) - tamanioVectorCaracteristicas - paso) + 1 ):

        input = serie[ i: i + tamanioVectorCaracteristicas]

        target = serie[ i + tamanioVectorCaracteristicas + paso - 1]

        inp.append(input)

        tar.append(target)

    return inp, tar


