import collections
from typing import Iterator


def consume(iterator: Iterator) -> None:
    collections.deque(iterator, maxlen=0)
