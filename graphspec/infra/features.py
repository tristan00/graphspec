import hashlib
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=200000)
def token(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def encoder(width, numbers, categories):
    allowed = frozenset(numbers + categories)

    @lru_cache(maxsize=250000)
    def encoded(kind, fields):
        data = dict(fields)
        values = np.zeros(width, np.float32)
        present = np.zeros(width, np.float32)
        for field in numbers:
            value = data.get(field)
            if isinstance(value, (int, float)) and np.isfinite(value):
                value = float(value)
                code = token(kind + "." + field)
                values[code % width] += (1 if code & 128 else -1) * np.sign(value) * np.log1p(abs(value))
                present[code % width] = 1
        for field in categories:
            value = data.get(field)
            if isinstance(value, (str, int)):
                code = token(kind + "." + field + "=" + str(value))
                values[code % width] += 1 if code & 128 else -1
                present[code % width] = 1
        code = token("kind=" + kind)
        values[code % width] += 1
        present[code % width] = 1
        return np.concatenate((np.sign(values) * np.log1p(np.abs(values)), present)).astype(np.float16)

    def encode(kind, data):
        fields = tuple(
            sorted(
                (key, value)
                for key, value in data.items()
                if key in allowed and isinstance(value, (str, int, float))
            )
        )
        return encoded(kind, fields)

    return encode
