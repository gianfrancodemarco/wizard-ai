import os
from typing import Annotated

import redis
from fastapi import Depends

REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')


def get_redis_client():
    return redis.Redis(password=REDIS_PASSWORD)


RedisClient = Annotated[redis.Redis, Depends(get_redis_client)]