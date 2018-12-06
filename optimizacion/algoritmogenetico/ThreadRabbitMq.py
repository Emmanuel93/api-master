import time, threading, pika

class ThreadRabbitMq(threading.Thread):
    def receive(self, ch, method, properties, body):
        print ' [x] received %r' % (body,)
        ##time.sleep( body.count('.') )
        ch.basic_ack(delivery_tag = method.delivery_tag)
        self.callback2(body)

    def __init__(self, host, username, password, queque, 
        callback2, action='consumer',durable=False, message=None):
        threading.Thread.__init__(self)
        self.action = action
        self.message = message
        self.queque = queque
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queque, durable=durable)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.receive, queue=queque)
        self.callback2 = callback2
        self.rabbit_queue = self.channel.queue_declare(queue=queque, durable=True, passive=True)

    def publish(self):
        self.channel.basic_publish(exchange='',
            routing_key=self.queque,
            body=self.message)
        
        self.connection.close()

    def get_messages_in_queque(self):
        print self.rabbit_queue.method.message_count

    def run(self):
        if self.action == 'consumer':
            print 'start consuming on queque of ' + self.queque
            self.channel.start_consuming()
        elif self.action == 'publisher':
            print 'publish message '+ self.message
            self.publish()
'''
print 'launch thread'
td = ThreadRabbitMq('localhost', 'guest', 'guest', 'hola2',callback2)
td.start()

print 'launch thread'
td = ThreadRabbitMq('localhost', 'guest', 'guest', 'hola3',callback2)
td.start()
'''