from __future__ import annotations
from CGRtools.algorithms.morgan import tuple_hash
from collections import defaultdict, deque
from math import log2
from numpy import zeros
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Collection, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer


class MorganFingerprint(TransformerMixin, BaseEstimator):
    def __init__(self, radius: int = 4, length: int = 1024):
        self._radius = radius
        self._mask = length - 1
        self._length = length
        self._log = int(log2(length))

    def transform(self, x: Collection):
        fingerprints = zeros((len(x), self._length))

        for idx, mol in enumerate(x):
            fingerprint = fingerprints[idx]
            arr = self._fragments(self._bfs(mol), mol)
            hashes = {tuple_hash(tpl) for tpl in arr}

            for one in hashes:
                fingerprint[one & self._mask - 1] = 1
                fingerprint[one >> self._log & self._mask - 1] = 1

        return fingerprints

    def _bfs(self, molecule: MoleculeContainer) -> list[list[int]]:
        atoms = molecule._atoms
        bonds = molecule._bonds

        arr = [[x] for x in atoms]
        queue = deque(arr)
        while queue:
            now = queue.popleft()
            if len(now) >= self._radius:
                continue
            var = [now + [x] for x in bonds[now[-1]] if x not in now]
            arr.extend(var)
            queue.extend(var)
        return arr

    def _fragments(self, arr: list[list], molecule: MoleculeContainer) -> set[tuple[Union[int, Any], ...]]:
        atoms = {x: int(a) for x, a in molecule.atoms()}
        bonds = molecule._bonds
        cache = defaultdict(dict)
        out = set()
        for frag in arr:
            var = [atoms[frag[0]]]
            for x, y in zip(frag, frag[1:]):
                b = cache[x].get(y)
                if not b:
                    b = cache[x][y] = cache[y][x] = int(bonds[x][y])
                var.append(b)
                var.append(atoms[y])
            var = tuple(var)
            rev_var = var[::-1]
            if var > rev_var:
                out.add(var)
            else:
                out.add(rev_var)
        return out


__all__ = ['MorganFingerprint']
