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
from ThreadRabbitMq import ThreadRabbitMq


class Solution(object):
    """
    A solution for the given problem, it is composed of a binary value and its fitness value
    """
    def __init__(self, dict):
        vars(self).update( dict )

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.value)


def generate_candidate(vector):
    """
    Generates a new candidate solution based on the probability vector
    """
    value = ""

    for p in vector:
        value += "1" if random() < p else "0"

    dictionario = {
        "value" : value,
        "fitness" : 0
    }

    return Solution(dictionario)


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

def recieve_individuos(individuo):
    global stop
    global contadorIndivudosEntrenados
    global IndividuosEntrenadoUno
    global IndividuosEntrenadoDos

    individuo = json.loads( individuo, object_hook= Solution )

    print('Recibiendo individuo ',individuo)
    print('individuo.value ',individuo.value)
    print('Recibiendo individuo ',individuo.fitness)

    contadorIndivudosEntrenados= contadorIndivudosEntrenados + 1

    print contadorIndivudosEntrenados
    if contadorIndivudosEntrenados == 2:
        IndividuosEntrenadoDos = individuo
        stop = False
        contadorIndivudosEntrenados = 0
    else:    
        IndividuosEntrenadoUno = individuo

def run(generations, size, population_size, fitness_function):
    global stop
    global vector
    global IndividuosEntrenadoUno
    global IndividuosEntrenadoDos
    # this is the probability for each solution bit be 1
    vector = generate_vector(size)
    # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    best = None

    # I stop by the number of generations but you can define any stop param
    for i in xrange(generations):
        stop = True
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
        print s1j
        print s2j

        channel.basic_publish(exchange='',
                              routing_key='individuos',
                              body=s1j)
        
        channel.basic_publish(exchange='',
                              routing_key='individuos',
                              body=s2j)
        connection.close()

        while(stop):
            a = 1
        # calculate fitness for each
        # s1.calculate_fitness(fitness_function)
        # s2.calculate_fitness(fitness_function)

        # let them compete, so we can know who is the best of the pair
        print "Inicia Competencia"
        winner, loser = compete(IndividuosEntrenadoUno, IndividuosEntrenadoDos)
        print "Finaliza Competencia"
        if best:
            if winner.fitness > best.fitness:
                best = winner
        else:
            best = winner

        print "generation: %d best value: %s best fitness: %f" % (i + 1, best.value, float(best.fitness))

        # updates the probability vector based on the success of each bit
        update_vector(vector, winner.value, loser.value, population_size)
        
    ##return self.mejorModelo
def callback2(body):
    global stop
    print body
    stop = False


if __name__ == '__main__':

    horaInicio = time.strftime("%H:%M:%S")
    fechaInicio = time.strftime("%d-%m-%Y")

    global numeroPaso
    global serie
    global mejorError
    global mejorModelo
    global mejorConfiguracion
    global contadorSolucion
    global stop
    global vector
    global IndividuosEntrenadoUno
    global IndividuosEntrenadoDos
    global contadorIndivudosEntrenados

    IndividuosEntrenadoUno = None
    IndividuosEntrenadoDos = None
    contadorIndivudosEntrenados = 0 

    stop = False
    vector = None

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

    print 'Lanzando escuchador de cuando los individuos ya han finalizado'
    td = ThreadRabbitMq('localhost', 'guest', 'guest', 'individuosEntrenados',recieve_individuos, durable=True)
    td.start()

    for numeroPaso in range(1, 2):

        mejorError = 99999999
        mejorModelo = None
        mejorConfiguracion = None

        f = lambda x: fitness(x)
        run(100, 19, 100, f)

        print "<<<<<<<<<<<<<<<<<<<<<  MEJOR CONFIGURACION PASO ", numeroPaso," >>>>>>>>>>>>>>>>"

        print mejorConfiguracion

        #mejorModelo.model.save("mejorModeloPaso"+str(numeroPaso)+".h5")

        #nombreIteracion = "paso"+str(numeroPaso)

        #helper.guardarMSEContinuoExcel("Experimento_resultadosMSE"+fechaInicio+"_"+horaInicio+".csv", nombreIteracion, mejorError, 'MSE_TEST', mejorConfiguracion, 'MEJOR_CONFIGURACION')

    # Hora y fecha inicio
    horaFin = time.strftime("%H:%M:%S")
    fechaFin = time.strftime("%d/%m/%Y")

    print "Inicio: ", fechaInicio," ", horaInicio, " termino: ",fechaFin, " ", horaFin
