from CIMtools.base import CIMtoolsTransformerMixin
from numba import njit, prange
from numpy import append, array, cumprod, ndenumerate, ones, sum as npsum, uint64, where, zeros


def convert_to_matrix(molecule):
    atoms = array([v for k, v in molecule.atoms()], dtype=uint64)
    adj = zeros((len(atoms) + 1, len(atoms) + 1), dtype=uint64)
    for k, v, b in molecule.bonds():
        adj[k][v] = adj[v][k] = b.order
    return atoms, adj


def bfs_pure_python(molecule, length):
    atoms = {k: v for k, v in molecule.atoms()}
    bonds = {k: {} for k in atoms}
    for k, v, b in molecule.bonds():
        bonds[k][v] = bonds[v][k] = b.order

    arr = []
    for atom in atoms:
        arr.append([atom])
        start, end = len(arr) - 1, len(arr)
        while True:
            for i in range(start, end):
                path, last = arr[i], arr[i][-1]
                for_adding = [path + [x] for x in bonds[last] if x not in path]
                arr.extend(for_adding)
            if len(arr[-1]) == length:
                break
            start, end = end, len(arr)
    return arr


@njit
def bfs_numba(atoms, adj, length):
    threes = array([4] + [3] * (length - 1))
    max_frags = len(atoms) * npsum(cumprod(threes))
    arr = zeros((max_frags, length), dtype=uint64)
    index = 0
    for i in range(1, len(atoms) + 1):
        arr[index][0] = i
        start, end = index, index + 1
        while True:
            for j in range(start, end):
                path = arr[j]
                if where(path == 0)[0].size == 0:
                    index += 1
                    break
                border = where(path != 0)[0][-1]
                last = path[border]
                for k in range(1, len(atoms) + 1):
                    if adj[last][k] and where(path == k)[0].size == 0:
                        path[border + 1] = k
                        index += 1
                        arr[index] = path
            if where(path == 0)[0].size == 0:
                index += 1
                break
            start, end = end, index + 1
    return arr


@njit(parallel=True)
def bfs_numba_parallel(atoms, adj, length):
    threes = array([4] + [3] * (length - 1))
    max_frags = len(atoms) * npsum(cumprod(threes))
    divider = max_frags // len(atoms)
    arr = zeros((max_frags, length), dtype=uint64)
    for i in prange(1, len(atoms) + 1):
        index = divider * (i - 1)
        arr[index][0] = i
        start, end = index, index + 1
        while True:
            for j in range(start, end):
                path = arr[j]
                if where(path == 0)[0].size == 0:
                    index += 1
                    break
                border = where(path != 0)[0][-1]
                last = path[border]
                for k in range(1, len(atoms) + 1):
                    if adj[last][k] and where(path == k)[0].size == 0:
                        path[border + 1] = k
                        index += 1
                        arr[index] = path
            if where(path == 0)[0].size == 0:
                index += 1
                break
            start, end = end, index + 1
    return arr


@njit(parallel=True)
def dfs(atoms, adj, length):
    # fixme: nonetype has no len error if it runs with prange and njit's parallel flag
    max_number = len(atoms) * 4 * 3**(length - 1)
    fragments = zeros((max_number, len(atoms)), dtype=uint64)
    index = 0
    for i, x in ndenumerate(atoms):
        stack = array([(i[0] + 1,), (0,)], dtype=uint64)
        path = [uint64(x) for x in range(0)]
        while stack.size > 0:
            not_seen = ones(len(atoms) + 1, dtype=uint64)
            for was in path:
                not_seen[was] = 0
            now, depth = stack[0][-1], stack[1][-1]
            old_stack = stack
            stack = zeros((len(stack), len(stack[0]) - 1), dtype=uint64)
            for k in range(len(old_stack)):
                stack[k] = old_stack[k][:-1]
            if len(path) > depth:
                path = path[:depth]
            path.append(now)
            if len(path) <= length:
                print(path)
                fragments[index][:len(path)] = array(path, dtype=uint64)
                index = index + 1
            depth += 1
            if depth + 1 > length:
                continue
            for z, b in ndenumerate(adj[now]):
                if not_seen[z[0]] and b:
                    stack = append(stack, array([(z[0],), (depth,)], dtype=uint64), axis=1)
    return fragments


def tuple_hash(v):
    """
    Python 3.8 hash for tuples implemented on python.
    Working only for nonnested tuples.
    """
    acc = 0x27D4EB2F165667C5
    for el in v:
        acc += el * 0xC2B2AE3D27D4EB4F
        acc = (acc << 31) | (acc >> 33)
        acc *= 0x9E3779B185EBCA87

    acc += len(v) ^ 2870177450013471926  # 0xC2B2AE3D27D4EB4F ^ 3527539

    if acc == 0xFFFFFFFFFFFFFFFF:
        return 1546275796
    return acc

