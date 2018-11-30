from __future__ import print_function
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='prueba',  durable=True)

channel.basic_publish(exchange='',
                      routing_key='prueba',
                      body='{ test : "test" , numero:2}')
print(" [x] Sent 'Hello World!'")
connection.close()