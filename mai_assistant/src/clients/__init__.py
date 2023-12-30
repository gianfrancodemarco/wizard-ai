"""
Clients to interact with external services
Dep are FastAPI dependencies that can be injected into FastAPI routes
"""

from .redis import get_redis_client, RedisClientDep
from .google import GoogleClient
from .rabbitmq import get_rabbitmq_producer, get_rabbitmq_consumer, RabbitMQConsumer, RabbitMQProducer, RabbitMQProducerDep