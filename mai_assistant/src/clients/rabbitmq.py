import logging
import os
from typing import Annotated

import pika
from fastapi import Depends

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST')
RABBITMQ_PORT = os.environ.get('RABBITMQ_PORT', 5672)
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD')
RABBITMQ_USER = os.environ.get('RABBITMQ_USER')

# Convert port to int if it is a string (Due to the fact that Kubernetes automatically populates some env variables from the services)
if type(RABBITMQ_PORT) == str and ':' in RABBITMQ_PORT:
    RABBITMQ_PORT = int(RABBITMQ_PORT.split(':')[-1])

logger = logging.getLogger(__name__)


class _RabbitMQClient:

    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def connect(self):
        connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=pika.PlainCredentials(self.user, self.password)
        )
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        return channel

    def publish(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        queue: str,
        message: str
    ):
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_publish(exchange='', routing_key=queue, body=message)

    def consume(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        queue: str,
        callback: callable
    ):
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
        channel.start_consuming()


def get_rabbitmq_client():
    return _RabbitMQClient(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        user=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD
    )


RabbitMQClient = Annotated[_RabbitMQClient, Depends(get_rabbitmq_client)]