from __future__ import print_function
import pika
import time
import json

def x(ch, method, properties, body):
    print(" [x] Received %r" % body)
    time.sleep(5)
    print(" [x] Done")
    ch.basic_ack(delivery_tag = method.delivery_tag)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.basic_qos(prefetch_count=2)
channel.basic_consume(x, queue='individuos', no_ack=True)
channel.start_consuming()
connection.close()
