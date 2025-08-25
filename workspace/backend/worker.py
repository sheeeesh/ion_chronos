import os
from rq import Connection, Queue, Worker
import redis

redis_url = os.getenv("REDIS_URL","redis://redis:6379/0")
conn = redis.from_url(redis_url)
queue = Queue("default", connection=conn)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(["default"])
        worker.work()
