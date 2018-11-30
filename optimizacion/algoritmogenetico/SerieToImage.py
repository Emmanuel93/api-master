from PIL import Image
import random
import csv
import math
import numpy as np
import cPickle
import Helper as helper
import pandas as pd
from sklearn.model_selection import train_test_split


def serieAImagen(serieOriginal, altoImagen, anchoImagen):

    serie = helper.normalizarSerieParaImagen(serieOriginal)

    lenSerie = len(serie)

    tamanioLadoImagen = int(math.sqrt(lenSerie))

    areaImagen = int(math.pow(tamanioLadoImagen, 2))

    serie = serie[:areaImagen]

    imageArray = []

    for value in serie:
        imageArray.append(int(value))

    datosImagen = np.array(imageArray).reshape(tamanioLadoImagen, tamanioLadoImagen)

    return datosImagen

def serieAImagen2(serieOriginal):

    imageArray = []

    for valor1 in serieOriginal:
        for valor2 in serieOriginal:

            ac = math.cos(valor1 + valor2)

            imageArray.append(ac)

    imageArray = helper.normalizarSerieParaImagen(imageArray)

    datosImagen = np.array(imageArray).reshape(len(serieOriginal), len(serieOriginal))

    return datosImagen


def obtenerBancoDeImagenes2(tamanioImagen, serieTiempo, paso):

    arrayInputTarget = []

    for i in range (0, (len(serieTiempo) - tamanioImagen - paso) + 1 ):

        input = serieTiempo[ i: i + tamanioImagen]

        input = helper.normalizarSerieParaImagen2(input)

        target = serieTiempo[ i + tamanioImagen + paso - 1]

        input = map(lambda x : math.acos(x), input)

        datosImagen = serieAImagen2(input)

        target = np.array(target)

        arrayInputTarget.append({'imagen' : datosImagen, 'target' : target})

    return arrayInputTarget


def obtenerBancoDeImagenesGAFCUDA(tamanioImagen, serieTiempo, paso):

    obtenerBancoDeImagenesGAFCUDA = helper.obtenerFuncionKernel("obtenerBancoDeImagenesGAFCUDA")

    n_elem = tamanioImagen
    block_size = 32
    n_blocks = (n_elem + block_size - 1) / block_size;
    block_conf = (block_size, 1, 1)
    grid_conf = (n_blocks, 1)

    lenSerieTiempo = len(serieTiempo)

    obtenerBancoDeImagenesGAFCUDA(cuda.InOut(serieTiempo), np.float32(lenSerieTiempo), np.float32(tamanioImagen), np.float32(paso), block = block_conf, grid = grid_conf)

    '''
    arrayInputTarget = []

    for i in range (0, (len(serieTiempo) - tamanioImagen - paso) + 1 ):

        input = serieTiempo[ i: i + tamanioImagen]

        min_value = min(input)
        max_value = max(input)

        input = np.asarray(input, dtype = np.float32)

        normalizarEntreMenusUnoYUno(np.int32(n_elem), np.float32(min_value), np.float32(max_value), cuda.InOut(input), block = block_conf, grid = grid_conf)

        target = serieTiempo[ i + tamanioImagen + paso - 1]

        input = map(lambda x : math.acos(x), input)

        datosImagen = serieAImagen2(input)

        target = np.array(target)

        arrayInputTarget.append({'imagen' : datosImagen, 'target' : target})
    '''

    return arrayInputTarget


def obtenerBancoDeImagenes(altoImagen, anchoImagen, serieTiempo, paso):

    areaImagen = altoImagen * anchoImagen

    arrayInputTarget = []

    for i in range (0, len(serieTiempo) - areaImagen ):

        input = serieTiempo[ i: i + areaImagen]
        target = serieTiempo[ i + areaImagen]

        datosImagen = serieAImagen(input, altoImagen, anchoImagen)

        arrayInputTarget.append({'imagen' : datosImagen, 'target' : target})

    return arrayInputTarget

def datosAImagen(bancoDeImagenes, target, directorioBancoImagenes):

    numeroImagen = 0

    relaciones = []
    for datos in bancoDeImagenes:

        altoImagen = datos.shape[0]
        anchoImagen = datos.shape[1]

        imagen = Image.new("L", (altoImagen, anchoImagen))
        imagen.putdata(datos.flatten())

        nombreImagen = str(numeroImagen+1) + '.jpeg'

        imagen.save(directorioBancoImagenes+"/"+nombreImagen,"JPEG")

        relaciones.append([nombreImagen, target[numeroImagen][0]])

        numeroImagen+=1

    df = pd.DataFrame(relaciones)
    df.to_csv(directorioBancoImagenes+"/relacion.csv")

def mostrarImagen(datosImagen):

    altoImagen = datosImagen.shape[0]
    anchoImagen = datosImagen.shape[1]

    imagen = Image.new("L", (altoImagen, anchoImagen))
    imagen.putdata(datosImagen.flatten())
    imagen.show()

def rotarImagen(datosImagen, grados):

    altoImagen = datosImagen.shape[0]
    anchoImagen = datosImagen.shape[1]

    imagen = Image.new("L", (altoImagen, anchoImagen))
    imagen.putdata(datosImagen.flatten())

    imagen = imagen.rotate(grados)

    return np.asarray(imagen)

def obtenerImagenes(serie, tipoSerieAImagen, numeroPaso, tamanioImagen, trainingRate):

    validationSize = 20 # 20 porciendo del total de los datos

    testSizeTotal = int(len(serie) * (validationSize / 100.0))

    #Los tipoSerieAImagen son ESCANEO_LINEAL y GRAMIAN_ANGULAR_FIELD

    # Datos de entrenamiento = 80 % de la serie original
    data_train_partial, data_test = train_test_split(serie, test_size=testSizeTotal, shuffle=False)

    trainigOffSize = 1 - trainingRate

    trainOffSizeTotal =  int( len(data_train_partial) * trainigOffSize)

    data_train, data_test_off = train_test_split(data_train_partial, test_size=trainOffSizeTotal, shuffle=False)

    # obtener el banco de imagenes para entrenamiento
    if tipoSerieAImagen == 'ESCANEO_LINEAL':
        bancoImagenesTrain = obtenerBancoDeImagenes(tamanioImagen, tamanioImagen, data_train)
    elif tipoSerieAImagen == 'GRAMIAN_ANGULAR_FIELD':
        bancoImagenesTrain = obtenerBancoDeImagenes2(tamanioImagen, data_train, numeroPaso)

    datosImagenesTrain, datosImagenesTrainTarget = obtenerDataSet(bancoImagenesTrain)

    datosImagenesTrain = np.array(datosImagenesTrain)
    datosImagenesTrainTarget = np.array(datosImagenesTrainTarget)

    #datosImagenesTrain, datosImagenesTrainTarget = rotar(datosImagenesTrain, datosImagenesTrainTarget, 90)

    # obtener el banco de imagenes para pruebas
    if tipoSerieAImagen == 'ESCANEO_LINEAL':
        bancoImagenesTest = obtenerBancoDeImagenes(tamanioImagen, tamanioImagen, data_test)
    elif tipoSerieAImagen == 'GRAMIAN_ANGULAR_FIELD':
        bancoImagenesTest = obtenerBancoDeImagenes2(tamanioImagen, data_test, numeroPaso)

    datosImagenesTest, datosImagenesTestTarget = obtenerDataSet(bancoImagenesTest)

    datosImagenesTest = np.array(datosImagenesTest)
    datosImagenesTestTarget = np.array(datosImagenesTestTarget)

    #SerieToImage.mostrarImagen(datosImagenesTest[0])

    return datosImagenesTrain, datosImagenesTrainTarget, datosImagenesTest, datosImagenesTestTarget

def obtenerDataSet(datos):

    #datos = cPickle.load(open(dataSetNombre,"rb"))

    datosImagenes = []
    datosTarget = []

    for value in datos:
        datoImagen = value['imagen']
        datoTarget = value['target']

        datosImagenes.append(datoImagen)
        datosTarget.append(datoTarget)

    return datosImagenes, datosTarget

if __name__ == "__main__":

    nombreArchivo = "Agua_3068_normalizado"
    rutaArchivo = nombreArchivo+'.csv'
    serieOriginal = helper.leerArchivo(rutaArchivo)
   # serieOriginal = helper.normalizarSerie(serieOriginal)

    altoImagen = 24
    anchoImagen = 24

    datosImagenesTrain, datosImagenesTrainTarget, datosImagenesTest, datosImagenesTestTarget= obtenerImagenes(serieOriginal, 'GRAMIAN_ANGULAR_FIELD', 1, altoImagen, .60)

    #nombreDataSetImagenes = "dataset"+str(altoImagen)+"x"+str(anchoImagen)+"_"+str(len(serieOriginal))+".data"

    datosAImagen(datosImagenesTrain, datosImagenesTrainTarget, "SeriesHector/imagenes/training")

    datosAImagen(datosImagenesTest, datosImagenesTestTarget, "Series/imagenes/validation")

    '''
    cPickle.dump(arrayInputTarget, open(nombreDataSetImagenes,"wb"))

    '''
