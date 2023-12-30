import os

import pika
from pika import channel

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST')
RABBITMQ_PORT = os.environ.get('RABBITMQ_PORT', 5672)
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD')
RABBITMQ_USER = os.environ.get('RABBITMQ_USER')

# Convert port to int if it is a string (Due to the fact that Kubernetes automatically populates some env variables from the services)
if type(RABBITMQ_PORT) == str and ':' in RABBITMQ_PORT:
    RABBITMQ_PORT = int(RABBITMQ_PORT.split(':')[-1])


class AsyncPikaConsumer:

    def __init__(
        self,
        on_message_callback: callable,
        queue_name='your_queue_name',
    ):
        self._on_message_callback = on_message_callback
        self.queue_name = queue_name
        self.connection_params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
        )
        self.connection = None
        self.channel = None

    def on_message(self, ch, method, properties, body):
        print(f" [x] Received {body}")
        self._on_message_callback(body)
        # Add your processing logic here

    def on_connected(self, connection: pika.SelectConnection):
        connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, new_channel: channel.Channel):
        self.channel = new_channel
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.on_message,
            auto_ack=True,
        )

        print(
            f" [*] Waiting for messages on {self.queue_name}. To exit, press Ctrl+C")

    def run_consumer(self):
        self.connection = pika.SelectConnection(
            self.connection_params, on_open_callback=self.on_connected)
        self.connection.ioloop.start()
