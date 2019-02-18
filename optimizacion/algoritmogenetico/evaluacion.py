from __future__ import print_function
import pika
import time
import json
import numpy as np
import time
import CNN as CNN
import Helper as helper

class Solution(object):
    """
    A solution for the given problem, it is composed of a binary value and its fitness value
    """
    def __init__(self, dict):
      vars(self).update( dict )

    def calculate_fitness(self, fitness_function):
		self.fitness = fitness_function(self.value)
		print("Calculando fitness")

def compete(a, b):
    """
    Returns a tuple with the winner solution
    """
    if a.fitness > b.fitness:
        return a, b
    else:
        return b, a

def x(ch, method, properties, body):
	global cont
	global individuo1
	global individuo2

	cont = cont + 1	

	f = lambda x: fitness(x)

	individuo = json.loads( body, object_hook= Solution )
	individuo.calculate_fitness(f)
	ch.basic_ack(delivery_tag=method.delivery_tag)
	credentials = pika.PlainCredentials('server', 'emmanuel')
	connection = pika.BlockingConnection(pika.ConnectionParameters( host='192.168.1.162',credentials=credentials ))
	channel = connection.channel()
	channel.queue_declare(queue='individuosEntrenados', durable=True)
	individuoEntrenado = json.dumps(individuo.__dict__)

	channel.basic_publish(exchange='',
	                      routing_key='individuosEntrenados',
	                      body=individuoEntrenado)

	print(" [x] Done")

def fitness(x):

	global serie
	global mejorError
	global mejorModelo
	global mejorConfiguracion
	global numeroPaso
	global contadorSolucion

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
	#optimizer =   posiciones

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

	nombreArchivo = "Agua_3121"
	rutaArchivo = nombreArchivo+'.csv'
	serie = "Agua_3121"

	numeroPaso = 1
	horaInicio = time.strftime("%H-%M-%S")
	fechaInicio = time.strftime("%d-%m-%Y")
	score, model = CNN.experimento(serie, numEpocas, learningRate, trainingRate, optimizer, activation, filterSize, strides, padding, pool, valorDropout, numeroPaso)
	print(score)
	horaFin = time.strftime("%H:%M:%S")
	fechaFin = time.strftime("%d-%m-%Y")
	model.save("Modelo"+fechaInicio+"_"+horaInicio+"_"+fechaFin+"_"+horaFin+".h5")

	return score[1]

if __name__ == '__main__':

	global cont
	global individuo1
	global individuo2
	global numeroPaso
	global serie
	global mejorError
	global mejorModelo
	global mejorConfiguracion
	global contadorSolucion
	global op

	op = False

	cont = 0
	individuo1 = None
	individuo2 = None

	contadorSolucion = 0

	nombreArchivo = "Agua_3121"
	rutaArchivo = nombreArchivo+'.csv'
	serie = helper.leerArchivo(rutaArchivo)

	serie = helper.normalizarSerie(serie)

	numeroPaso = 1
	#serie = serie[:2000]

	print(nombreArchivo)

	credentials = pika.PlainCredentials('server', 'emmanuel')
	connection = pika.BlockingConnection(pika.ConnectionParameters( host='192.168.1.162',credentials=credentials, heartbeat_interval=65535, blocked_connection_timeout=65535))
	channel = connection.channel()
	channel.basic_qos(prefetch_count=1)
	channel.basic_consume(x, queue='individuos')
	channel.start_consuming()
