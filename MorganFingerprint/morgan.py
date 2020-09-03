from CIMtools.base import CIMtoolsTransformerMixin
from numba import njit, prange
from numpy import append, array, copy, cumprod, ndenumerate, ones, sum as npsum, uint64, where, zeros


def convert_to_matrix(molecule):
    atoms = array([v for k, v in molecule.atoms()], dtype=uint64)
    adj = zeros((len(atoms) + 1, len(atoms) + 1), dtype=uint64)
    for k, v, b in molecule.bonds():
        adj[k][v] = adj[v][k] = b.order
    return atoms, adj


def convert_to_another(molecule):
    atoms = array([v for k, v in molecule.atoms()], dtype=uint64)
    bonds = {x: [] for x, _ in molecule.atoms()}
    for k, v, b in molecule.bonds():
        bonds[k].append(v)
        bonds[v].append(k)
    max_len = len(max(bonds.values(), key=len))
    matrix = -ones((len(atoms), max_len))
    for k in bonds:
        matrix[k - 1][:len(bonds[k])] = array(bonds[k], dtype=uint64)
    return atoms, matrix


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
                path = copy(arr[j])
                where_flag = False
                for number in path:
                    if number == 0:
                        where_flag = True
                if where_flag:
                    index += 1
                    break
                border = 0
                for idx, number in enumerate(path):
                    if number != 0:
                        border = idx
                        break
                last = int(path[border])
                for k in adj[last - 1]:
                    where_flag = False
                    for number in path:
                        if number == 0:
                            where_flag = True
                    if k != -1 and not where_flag:
                        path[border + 1] = k
                        index += 1
                        arr[index] = path
            where_flag = False
            for number in path:
                if number == 0:
                    where_flag = True
            if where_flag:
                index += 1
                break
            start, end = end, index + 1
    return arr


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

