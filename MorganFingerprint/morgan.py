from numba import njit
from numpy import array, copy, ones, uint64, zeros
from sklearn.base import BaseEstimator, TransformerMixin


class MorganFingerprint(TransformerMixin, BaseEstimator):
    def __init__(self, radius=4, length=1024):
        self._atoms = {}
        self._bonds = {}
        self._radius = radius
        self._length = length

    def transform(self, x):
        arr = self._fragments(self._bfs(x))
        tuples = {self.tuple_hash(tpl) for tpl in arr}

        fingerprint = [0] * self._length
        for one in tuples:
            fingerprint[one & self._length - 1] = 1
            fingerprint[one >> 10 & self._length - 1] = 1

        return fingerprint

    def _bfs(self, molecule):
        self._atoms = {k: int(v) for k, v in molecule.atoms()}
        self._bonds = {k: {} for k in self._atoms}
        for k, v, b in molecule.bonds():
            self._bonds[k][v] = self._bonds[v][k] = b.order

        arr = []
        for atom in self._atoms:
            arr.append([atom])
            start, end = len(arr) - 1, len(arr)
            while True:
                for i in range(start, end):
                    path, last = arr[i], arr[i][-1]
                    for_adding = [path + [x] for x in self._bonds[last] if x not in path]
                    arr.extend(for_adding)
                if len(arr[-1]) == self._radius:
                    break
                start, end = end, len(arr)
        return arr

    def _fragments(self, arr):
        out = set()
        for frag in arr:
            frag_out = [self._atoms[frag[0]]]
            for first, second in zip(frag, frag[1:]):
                frag_out.extend([self._bonds[first][second], self._atoms[second]])
            frag_out = tuple(frag_out)
            if frag_out < frag_out[::-1]:
                out.add(frag_out)
            else:
                out.add(frag_out[::-1])
        return out

    def tuple_hash(self, v):
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


def mol_to_hard_adj(molecule):
    atoms = array([int(v) for k, v in molecule.atoms()], dtype=uint64)
    bonds = {x: [] for x, _ in molecule.atoms()}
    for k, v, b in molecule.bonds():
        bonds[k].append(v)
        bonds[v].append(k)
    max_len = len(max(bonds.values(), key=len))
    matrix = -ones((len(atoms), max_len))
    for k in bonds:
        matrix[k - 1][:len(bonds[k])] = array(bonds[k], dtype=uint64)
    return atoms, matrix


@njit
def bfs_numba(atoms, adj, length):
    max_frags = len(atoms) * 4 * 3**(length - 1)
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


__all__ = ['MorganFingerprint']
