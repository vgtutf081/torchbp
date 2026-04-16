from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EnqueuedJob:
    external_id: str


class QueueBackend:
    def enqueue(self, func: Callable, *args, **kwargs) -> EnqueuedJob:
        raise NotImplementedError


class InlineQueueBackend(QueueBackend):
    def enqueue(self, func: Callable, *args, **kwargs) -> EnqueuedJob:
        func(*args, **kwargs)
        return EnqueuedJob(external_id="inline")


class RQQueueBackend(QueueBackend):
    def __init__(self, redis_url: str) -> None:
        from redis import Redis
        from rq import Queue

        self._queue = Queue("torchbp", connection=Redis.from_url(redis_url))

    def enqueue(self, func: Callable, *args, **kwargs) -> EnqueuedJob:
        task = self._queue.enqueue(func, *args, **kwargs)
        return EnqueuedJob(external_id=task.id)
