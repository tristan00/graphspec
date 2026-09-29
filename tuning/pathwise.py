import math

import numpy as np
from numba import njit

GRID = 32
REACH = 4


@njit(cache=True)
def warp(a, b, u, penalty):
    n, m = len(a), len(b)
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(total[i - 1, j - 1], total[i - 1, j] + penalty, total[i, j - 1] + penalty)
            total[i, j] = abs(a[i - 1] - b[j - 1]) + best
    i, j, cells, distortion = n, m, 0, 0.0
    while i > 0 and j > 0:
        cells += 1
        distortion += (u[i - 1] - u[j - 1]) ** 2
        diagonal, up, left = total[i - 1, j - 1], total[i - 1, j] + penalty, total[i, j - 1] + penalty
        if diagonal <= up and diagonal <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    return total[n, m], cells, distortion


@njit(cache=True)
def soft_warp(a, b, gamma):
    n, m = len(a), len(b)
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            x, y, z = -total[i - 1, j - 1] / gamma, -total[i - 1, j] / gamma, -total[i, j - 1] / gamma
            top = max(x, y, z)
            soft = -gamma * (top + math.log(math.exp(x - top) + math.exp(y - top) + math.exp(z - top)))
            total[i, j] = (a[i - 1] - b[j - 1]) ** 2 + soft
    return total[n, m]


@njit(cache=True)
def soft_distortion(a, b, u, gamma):
    n, m = len(a), len(b)
    total = np.full((n + 2, m + 2), np.inf)
    total[0, 0] = 0.0
    cost = np.zeros((n + 2, m + 2))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = (a[i - 1] - b[j - 1]) ** 2
            x, y, z = -total[i - 1, j - 1] / gamma, -total[i - 1, j] / gamma, -total[i, j - 1] / gamma
            top = max(x, y, z)
            soft = -gamma * (top + math.log(math.exp(x - top) + math.exp(y - top) + math.exp(z - top)))
            total[i, j] = cost[i, j] + soft
    share = np.zeros((n + 2, m + 2))
    share[n + 1, m + 1] = 1.0
    total[n + 1, m + 1] = total[n, m]
    for i in range(1, n + 1):
        total[i, m + 1] = -np.inf
    for j in range(1, m + 1):
        total[n + 1, j] = -np.inf
    result = 0.0
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            down = math.exp((total[i + 1, j] - total[i, j] - cost[i + 1, j]) / gamma)
            right = math.exp((total[i, j + 1] - total[i, j] - cost[i, j + 1]) / gamma)
            across = math.exp((total[i + 1, j + 1] - total[i, j] - cost[i + 1, j + 1]) / gamma)
            share[i, j] = share[i + 1, j] * down + share[i, j + 1] * right + share[i + 1, j + 1] * across
            result += share[i, j] * (u[i - 1] - u[j - 1]) ** 2
    return result


@njit(cache=True)
def twed(a, b, u, stiffness, penalty):
    n, m = len(a), len(b)
    total = np.full((n, m), np.inf)
    total[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                best = min(
                    best, total[i - 1, j] + abs(a[i] - a[i - 1]) + stiffness * (u[i] - u[i - 1]) + penalty
                )
            if j > 0:
                best = min(
                    best, total[i, j - 1] + abs(b[j] - b[j - 1]) + stiffness * (u[j] - u[j - 1]) + penalty
                )
            if i > 0 and j > 0:
                best = min(
                    best,
                    total[i - 1, j - 1]
                    + abs(a[i] - b[j])
                    + abs(a[i - 1] - b[j - 1])
                    + stiffness * (abs(u[i] - u[j]) + abs(u[i - 1] - u[j - 1])),
                )
            total[i, j] = best
    return total[n - 1, m - 1]


@njit(cache=True)
def erp(a, b, gap):
    n, m = len(a), len(b)
    total = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        total[i, 0] = total[i - 1, 0] + abs(a[i - 1] - gap)
    for j in range(1, m + 1):
        total[0, j] = total[0, j - 1] + abs(b[j - 1] - gap)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            total[i, j] = min(
                total[i - 1, j - 1] + abs(a[i - 1] - b[j - 1]),
                total[i - 1, j] + abs(a[i - 1] - gap),
                total[i, j - 1] + abs(b[j - 1] - gap),
            )
    return total[n, m]


@njit(cache=True)
def edits(a, b, tolerance):
    n, m = len(a), len(b)
    total = np.zeros((n + 1, m + 1))
    common = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        total[i, 0] = i
    for j in range(1, m + 1):
        total[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            near = abs(a[i - 1] - b[j - 1]) <= tolerance
            total[i, j] = min(
                total[i - 1, j - 1] + (0.0 if near else 1.0), total[i - 1, j] + 1, total[i, j - 1] + 1
            )
            common[i, j] = common[i - 1, j - 1] + 1 if near else max(common[i - 1, j], common[i, j - 1])
    return total[n, m] / max(n, m), 1 - common[n, m] / min(n, m)


@njit(cache=True)
def move(new, neighbour, other, cost):
    if neighbour <= new <= other or neighbour >= new >= other:
        return cost
    return cost + min(abs(new - neighbour), abs(new - other))


@njit(cache=True)
def msm(a, b, cost):
    n, m = len(a), len(b)
    total = np.zeros((n, m))
    total[0, 0] = abs(a[0] - b[0])
    for i in range(1, n):
        total[i, 0] = total[i - 1, 0] + move(a[i], a[i - 1], b[0], cost)
    for j in range(1, m):
        total[0, j] = total[0, j - 1] + move(b[j], a[0], b[j - 1], cost)
    for i in range(1, n):
        for j in range(1, m):
            total[i, j] = min(
                total[i - 1, j - 1] + abs(a[i] - b[j]),
                total[i - 1, j] + move(a[i], a[i - 1], b[j], cost),
                total[i, j - 1] + move(b[j], a[i], b[j - 1], cost),
            )
    return total[n - 1, m - 1]


@njit(cache=True)
def curves(a, b, u):
    n = len(a)
    reach = np.empty((n, n))
    forward, backward = 0.0, 0.0
    nearest = np.full(n, np.inf)
    for i in range(n):
        closest = np.inf
        for j in range(n):
            gap = math.sqrt((u[i] - u[j]) ** 2 + (a[i] - b[j]) ** 2)
            closest = min(closest, gap)
            nearest[j] = min(nearest[j], gap)
            if i == 0 and j == 0:
                reach[i, j] = gap
            elif i == 0:
                reach[i, j] = max(reach[i, j - 1], gap)
            elif j == 0:
                reach[i, j] = max(reach[i - 1, j], gap)
            else:
                reach[i, j] = max(min(reach[i - 1, j], reach[i - 1, j - 1], reach[i, j - 1]), gap)
        forward = max(forward, closest)
    for j in range(n):
        backward = max(backward, nearest[j])
    return reach[n - 1, n - 1], max(forward, backward)


@njit(cache=True)
def signature(path, depth):
    n, d = path.shape
    sizes = np.empty(depth + 1, np.int64)
    sizes[0] = 1
    for k in range(1, depth + 1):
        sizes[k] = sizes[k - 1] * d
    offsets = np.zeros(depth + 2, np.int64)
    for k in range(depth + 1):
        offsets[k + 1] = offsets[k] + sizes[k]
    held = np.zeros(offsets[-1])
    held[0] = 1.0
    piece = np.zeros(offsets[-1])
    fresh = np.zeros(offsets[-1])
    for s in range(1, n):
        step = path[s] - path[s - 1]
        piece[:] = 0.0
        piece[0] = 1.0
        for k in range(1, depth + 1):
            for i in range(sizes[k - 1]):
                for c in range(d):
                    piece[offsets[k] + i * d + c] = piece[offsets[k - 1] + i] * step[c] / k
        fresh[:] = 0.0
        for k in range(depth + 1):
            for left in range(k + 1):
                right = k - left
                for i in range(sizes[left]):
                    value = held[offsets[left] + i]
                    if value != 0.0:
                        for j in range(sizes[right]):
                            fresh[offsets[k] + i * sizes[right] + j] += value * piece[offsets[right] + j]
        held[:] = fresh
    return held[1:]


@njit(cache=True)
def signature_kernel(x, y, refine):
    n, m = x.shape[0] - 1, y.shape[0] - 1
    parts = 1 << refine
    field = np.ones((n * parts + 1, m * parts + 1))
    for i in range(n * parts):
        for j in range(m * parts):
            inner = 0.0
            for c in range(x.shape[1]):
                inner += (x[i // parts + 1, c] - x[i // parts, c]) * (y[j // parts + 1, c] - y[j // parts, c])
            inner /= parts * parts
            field[i + 1, j + 1] = (field[i + 1, j] + field[i, j + 1]) * (
                1 + 0.5 * inner + inner * inner / 12
            ) - field[i, j] * (1 - inner * inner / 12)
    return field[-1, -1]


@njit(cache=True)
def resample(u, values, size):
    grid = np.linspace(u[0], u[-1], size)
    return np.interp(grid, u, values)


@njit(cache=True)
def shape_distance(a, b):
    n = len(a)
    scale = math.sqrt((a * a).sum() * (b * b).sum())
    if scale == 0:
        return np.nan
    best = -np.inf
    for shift in range(-(n - 1), n):
        total = 0.0
        for i in range(max(0, shift), min(n, n + shift)):
            total += a[i] * b[i - shift]
        best = max(best, total / scale)
    return 1 - best


@njit(cache=True)
def velocity(values):
    size = len(values) - 1
    result = np.empty(size)
    for i in range(size):
        slope = (values[i + 1] - values[i]) * size
        result[i] = math.copysign(math.sqrt(abs(slope)), slope)
    return result


@njit(cache=True)
def elastic(a, b):
    first, second = velocity(a), velocity(b)
    size = len(first)
    total = np.full((size + 1, size + 1), np.inf)
    total[0, 0] = 0.0
    origin_i = np.zeros((size + 1, size + 1), np.int64)
    origin_j = np.zeros((size + 1, size + 1), np.int64)
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            for di in range(1, min(REACH, i) + 1):
                for dj in range(1, min(REACH, j) + 1):
                    before = total[i - di, j - dj]
                    if before == np.inf:
                        continue
                    slope = dj / di
                    root = math.sqrt(slope)
                    cost = 0.0
                    for r in range(di):
                        position = (j - dj) + (r + 0.5) * slope
                        cost += (first[i - di + r] - second[min(int(position), size - 1)] * root) ** 2
                    cost = before + cost / size
                    if cost < total[i, j]:
                        total[i, j] = cost
                        origin_i[i, j], origin_j[i, j] = i - di, j - dj
    i, j, overlap = size, size, 0.0
    while i > 0:
        k, m = origin_i[i, j], origin_j[i, j]
        overlap += math.sqrt((j - m) / (i - k)) * (i - k) / size
        i, j = k, m
    return math.sqrt(total[size, size]), math.acos(min(1.0, overlap))


@njit(cache=True)
def variogram(a, b, u, power):
    n = len(a)
    total, weight = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            share = 1.0 / (u[j] - u[i])
            total += share * (abs(a[i] - a[j]) ** power - abs(b[i] - b[j]) ** power) ** 2
            weight += share
    return total / weight if weight > 0 else np.nan


@njit(cache=True)
def mass(a, b, waste):
    first, second, total = 0.0, 0.0, 0.0
    for i in range(len(a)):
        first += a[i]
        second += b[i]
        total += (waste if i == len(a) - 1 else 1.0) * abs(first - second)
    return total


@njit(cache=True)
def transport(a, b, waste):
    return mass(np.maximum(a, 0), np.maximum(b, 0), waste) + mass(np.maximum(-a, 0), np.maximum(-b, 0), waste)


@njit(cache=True)
def preserving(a, b, u, moment, entropy, width, rounds):
    n = len(a)
    span = u[-1] - u[0]
    kernel = np.empty((n, n))
    cost = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            gap = (u[i] - u[j]) / span
            cost[i, j] = abs(a[i] - b[j])
            prior = math.exp(-(gap * gap) / (2 * width * width)) / (width * math.sqrt(2 * math.pi))
            kernel[i, j] = prior * math.exp((moment / (gap * gap + 1) - moment - cost[i, j]) / entropy)
    left, right = np.ones(n), np.ones(n)
    for _ in range(rounds):
        left = (1.0 / n) / (kernel @ right)
        right = (1.0 / n) / (kernel.T @ left)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += left[i] * kernel[i, j] * right[j] * cost[i, j]
    return total


@njit(cache=True)
def staircase(values):
    size = len(values)
    result = np.empty((2 * size - 1, 2))
    for i in range(size):
        result[2 * i, 0] = values[i]
        result[2 * i, 1] = values[i]
        if i + 1 < size:
            result[2 * i + 1, 0] = values[i + 1]
            result[2 * i + 1, 1] = values[i]
    return result
