import os
import redis

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')

RedisClient = redis.Redis(
    host=REDIS_HOST,
    password=REDIS_PASSWORD
)