# "THE BEER-WARE LICENSE" (Revision 42):
# <cmte.igor.almeida@gmail.com> wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return.

__author__ = 'igor'

from random import random
import Helper as helper
import time
import numpy as np
import pika
import json


class Solution(object):
    """
    A solution for the given problem, it is composed of a binary value and its fitness value
    """
    def __init__(self, value):
        self.value = value
        self.fitness = 0

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.value)


def generate_candidate(vector):
    """
    Generates a new candidate solution based on the probability vector
    """
    value = ""

    for p in vector:
        value += "1" if random() < p else "0"

    return Solution(value)


def generate_vector(size):
    """
    Initializes a probability vector with given size
    """
    return [0.5] * size


def compete(a, b):
    """
    Returns a tuple with the winner solution
    """
    if a.fitness > b.fitness:
        return a, b
    else:
        return b, a


def update_vector(vector, winner, loser, population_size):
    '''
        individuo 0111000101100001110
        individuo 0001011010110001000
        '''
    for i in xrange(len(vector)):
        if winner[i] != loser[i]:
            if winner[i] == '1':
                vector[i] += 1.0 / float(population_size)
            else:
                vector[i] -= 1.0 / float(population_size)


def run(generations, size, population_size, fitness_function):
    # this is the probability for each solution bit be 1
    vector = generate_vector(size)
    # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    best = None

    # I stop by the number of generations but you can define any stop param
    for i in xrange(generations):

        print "GENERACION : ", i
        # generate two candidate solutions, it is like the selection on a conventional GA
        s1 = generate_candidate(vector)
        s2 = generate_candidate(vector)

        '''
        individuo 0111000101100001110
        individuo 0001011010110001000
        '''
        
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()

        channel.queue_declare(queue='individuos',  durable=True)

        s1j = json.dumps(s1.__dict__) # {"value": "0110001111110111101", "fitness": 0}
        s2j = json.dumps(s2.__dict__) # {"value": "0110001111110111101", "fitness": 0}

        channel.basic_publish(exchange='',
                              routing_key='individuos',
                              body=s1j)
        
        channel.basic_publish(exchange='',
                              routing_key='individuos',
                              body=s2j)
        connection.close()

        # calculate fitness for each
        # s1.calculate_fitness(fitness_function)
        # s2.calculate_fitness(fitness_function)

        # let them compete, so we can know who is the best of the pair
        #winner, loser = compete(s1, s2)

        '''if best:
            if winner.fitness > best.fitness:
                best = winner
        else:
            best = winner

        # updates the probability vector based on the success of each bit
        update_vector(vector, winner.value, loser.value, population_size)
        '''
    ##return self.mejorModelo
    #print "generation: %d best value: %s best fitness: %f" % (i + 1, best.value, float(best.fitness))

'''
def fitness(x):

    global serie
    global mejorError
    global mejorModelo
    global mejorConfiguracion
    global numeroPaso
    global contadorSolucion

    print "contadorSolucion:", contadorSolucion

    contadorSolucion = contadorSolucion + 1

    castigo = -9999999

    # 0 no es valido
    #epocas 20, 30, 40, 50, 60, 70, 80, 90 ... 140
    #maximo binario 111 = 7
    #numEpochs = 3 posiciones
    numEpocas = x[:3]
    numEpocas = helper.binarioADecimal(numEpocas)

    numEpocas = (numEpocas + 1) * 20

    #lr 0.0001, 0.0006, 0.0011, 0.0016, 0.0021, 0.0026, 0.0031
    #maximo binario 111 = 7
    #learningRate = 3 posiciones

    learningRate = x[3:6]
    learningRate = helper.binarioADecimal(learningRate)

    if learningRate   == 0:
        learningRate = 0.0001
    elif learningRate == 1:
        learningRate = 0.0006
    elif learningRate == 2:
        learningRate = 0.0011
    elif learningRate == 3:
        learningRate = 0.0016
    elif learningRate == 4:
        learningRate = 0.0021
    elif learningRate == 5:
        learningRate = 0.0026
    elif learningRate == 6:
        learningRate = 0.0031
    elif learningRate == 7:
        learningRate = 0.0036

    #training rate 0.60 0.70, 0.80, 0.90 porciento
    #maximo binario 11 = 3
    #trainingRate = 2 posiciones

    trainingRate = x[6:8]
    trainingRate = helper.binarioADecimal(trainingRate)

    trainingRate = trainingRate + 1

    if trainingRate   == 1:
        trainingRate = 0.70
    elif trainingRate == 2:
        trainingRate = 0.80
    elif trainingRate == 3:
        trainingRate = 0.90
    elif trainingRate == 4:
        trainingRate = 1

    # 0 no es valido
    #training GradientDescentOptimizer, AdamOptimizer and RMSPropOptimizero
    #maximo binario 11 = 3
    #optimizer = 2 posiciones

    optimizer = x[8:10]
    optimizer = helper.binarioADecimal(optimizer)

    if optimizer   == 0:
        return castigo
    elif optimizer == 1:
        optimizer = 'SGD'
    elif optimizer == 2:
        optimizer = 'ADAM'
    elif optimizer == 3:
        optimizer = 'RMSprop'

    # 0 no es valido
    # 3 no es valido
    #activation relu elu
    #maximo binario 1 = 1
    #activation = 1 posiciones
    activation = x[10]
    activation = helper.binarioADecimal(activation)

    if activation == 0:
        activation = 'relu'
    elif activation == 1:
        activation = 'elu'

    # 0 no es valido
    #filterSizes 3,4,5,6
    #maximo binario 11 = 3
    #filterSize = 2 posiciones
    filterSize = x[11:13]
    filterSize = helper.binarioADecimal(filterSize)

    filterSize = filterSize + 3

    # 0 no es valido
    #strides 1,2,3,4
    #maximo binario 11 = 3
    #strides = 2 posiciones
    strides = x[13:15]
    strides = helper.binarioADecimal(strides)

    strides = strides + 1

    #padding valid, same
    #maximo binario 1 = 1
    #padding  = 1 posicion
    padding = x[15]
    padding = helper.binarioADecimal(padding)

    if padding == 0:
        padding = 'valid'
    else :
        padding = 'same'

    #pooling max_pooling, avg_pooling
    #maximo binario 1 = 1
    #pool = 1
    pool = x[16]
    pool = helper.binarioADecimal(pool)

    if pool == 0:
        pool = 'MaxPooling2D'
    elif pool == 1:
        pool = 'AveragePooling2D'

    valorDropout = x[17:]

    valorDropout = helper.binarioADecimal(valorDropout)

    if valorDropout == 0:
        valorDropout = 0.3
    elif valorDropout == 1:
        valorDropout = 0.4
    elif valorDropout == 2:
        valorDropout = 0.5
    elif valorDropout == 3:
        valorDropout = 0.6

    error, y_test, prediccionEnTest, model, y_test, prediccionEnTest = CNN.experimento(serie, numEpocas, learningRate, trainingRate, optimizer, activation, filterSize, strides, padding, pool, valorDropout, numeroPaso)

    if error < mejorError:
        mejorError = error
        mejorModelo = model
        mejorConfiguracion = "numEpocas:", numEpocas, "-learningRate:",learningRate,"-trainingRate:",trainingRate,"-optimizer:",optimizer,"-activation:",activation,"-filterSize: ",filterSize,"-strides: ",strides,"-padding:",padding,"-pool:",pool,"-valorDropout:",valorDropout,"-numeroPaso:",numeroPaso

        #print "y_test: ", y_test
        #print "prediccionEnTest :", prediccionEnTest

        print "Mejor error obtenido >>>", error

    return error * -1
'''

if __name__ == '__main__':

    horaInicio = time.strftime("%H-%M-%S")
    fechaInicio = time.strftime("%d-%m-%Y")

    global numeroPaso
    global serie
    global mejorError
    global mejorModelo
    global mejorConfiguracion
    global contadorSolucion

    contadorSolucion = 0
    '''
    nombreArchivo = "wind_aristeomercado_10m_complete_21374-25374_suavizado"
    rutaArchivo = nombreArchivo+'.csv'
    serie = helper.leerArchivo(rutaArchivo)

    serie = helper.normalizarSerie(serie)

    print "serieNormailzadoa ", np.array(serie).reshape(len(serie), 1)

    #serie = serie[:2000]

    print nombreArchivo
    '''
    for numeroPaso in range(1, 2):

        mejorError = 99999999
        mejorModelo = None
        mejorConfiguracion = None

        f = lambda x: fitness(x)
        run(100, 19, 100, f)

        #print "<<<<<<<<<<<<<<<<<<<<<  MEJOR CONFIGURACION PASO ", numeroPaso," >>>>>>>>>>>>>>>>"

        #print mejorConfiguracion

        #mejorModelo.model.save("mejorModeloPaso"+str(numeroPaso)+".h5")

        #nombreIteracion = "paso"+str(numeroPaso)

        #helper.guardarMSEContinuoExcel(nombreArchivo+"_resultadosMSE"+fechaInicio+"_"+horaInicio+".csv", nombreIteracion, mejorError, 'MSE_TEST', mejorConfiguracion, 'MEJOR_CONFIGURACION')

    # Hora y fecha inicio
    horaFin = time.strftime("%H:%M:%S")
    fechaFin = time.strftime("%d/%m/%Y")

    print "Inicio: ", fechaInicio," ", horaInicio, " terminio: ",fechaFin, " ", horaFin
