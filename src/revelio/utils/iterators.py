import collections
from typing import Iterable, Iterator


def consume(iterator: Iterable | Iterator) -> None:
    collections.deque(iterator, maxlen=0)
