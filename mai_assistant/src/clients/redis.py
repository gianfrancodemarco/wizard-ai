import os
from typing import Annotated

import redis
from fastapi import Depends

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        password=REDIS_PASSWORD
    )


RedisClient = Annotated[redis.Redis, Depends(get_redis_client)]