"""Rate limiter abstraction with optional Redis backing."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

try:
    import redis
except Exception:  # pragma: no cover - optional dependency import guard
    redis = None

LOGGER = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, config: Any):
        self.limit = int(config.get("RATE_LIMIT_PER_MINUTE", 60))
        self.interval_seconds = 60
        self.redis_url = config.get("REDIS_URL", "")
        self._local_buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis_client = None

        if self.redis_url and redis is not None:
            try:
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self._redis_client.ping()
                LOGGER.info("Rate limiter using Redis backend")
            except Exception:
                LOGGER.exception("Redis unavailable; falling back to in-memory rate limiting")
                self._redis_client = None

    def is_allowed(self, client_id: str) -> bool:
        if self._redis_client is not None:
            return self._is_allowed_redis(client_id)
        return self._is_allowed_memory(client_id)

    def _is_allowed_memory(self, client_id: str) -> bool:
        now = time.time()
        bucket = self._local_buckets[client_id]

        while bucket and now - bucket[0] > self.interval_seconds:
            bucket.popleft()

        if len(bucket) >= self.limit:
            return False

        bucket.append(now)
        return True

    def _is_allowed_redis(self, client_id: str) -> bool:
        assert self._redis_client is not None
        now = time.time()
        window_start = now - self.interval_seconds
        key = f"rate:{client_id}"

        pipeline = self._redis_client.pipeline()
        pipeline.zremrangebyscore(key, 0, window_start)
        pipeline.zcard(key)
        _, count = pipeline.execute()

        if int(count) >= self.limit:
            return False

        pipeline = self._redis_client.pipeline()
        pipeline.zadd(key, {str(now): now})
        pipeline.expire(key, self.interval_seconds + 5)
        pipeline.execute()
        return True
