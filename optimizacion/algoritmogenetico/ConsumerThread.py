class ConsumerThread(threading.Thread):
    def __init__(self, host, *args, **kwargs):
        super(ConsumerThread, self).__init__(*args, **kwargs)

        self._host = host

    # Not necessarily a method.
    def callback_func(self, channel, method, properties, body):
        print("{} received '{}'".format(self.name, body))

    def run(self):
        credentials = pika.PlainCredentials("guest", "guest")

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self._host,
                                      credentials=credentials))

        channel = connection.channel()

        result = channel.queue_declare(exclusive=True)

        channel.queue_bind(result.method.queue,
                           exchange="my-exchange",
                           routing_key="*.*.*.*.*")

        channel.basic_consume(self.callback_func,
                              result.method.queue,
                              no_ack=True)

        channel.start_consuming()