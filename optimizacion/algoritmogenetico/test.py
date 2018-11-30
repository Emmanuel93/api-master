import time, threading, pika

class ThreadRabbitMq(threading.Thread):
    def receive(self, ch, method, properties, body):
        print ' [x] received %r' % (body,)
        ##time.sleep( body.count('.') )
        ch.basic_ack(delivery_tag = method.delivery_tag)
        self.callback2(body)

    def __init__(self, host, username, password, queque, callback2, durable=False):
        threading.Thread.__init__(self)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queque, durable=durable)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.receive, queue=queque)
        self.callback2 = callback2


    def run(self):
        print 'start consuming'
        self.channel.start_consuming()



def callback2(param):
    print param + "paquete"

print 'launch thread'
td = ThreadRabbitMq('localhost', 'guest', 'guest', 'individuos',callback2, True)
td.start()

print 'launch thread'
td = ThreadRabbitMq('localhost', 'guest', 'guest', 'hola2',callback2)
td.start()

print 'launch thread'
td = ThreadRabbitMq('localhost', 'guest', 'guest', 'hola3',callback2)
td.start()
