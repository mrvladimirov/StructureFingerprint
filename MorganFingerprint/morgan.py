# -*- coding: utf-8 -*-
#
#  Copyright 2020 Aleksandr Sizov <murkyrussian@gmail.com>
#  Copyright 2020 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of MorganFingerprint.
#
#  MorganFingerprint is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from CGRtools.algorithms.morgan import tuple_hash
from collections import defaultdict, deque
from math import log2
from numpy import zeros
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Collection, TYPE_CHECKING, List, Dict, Tuple

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer


class MorganFingerprint(TransformerMixin, BaseEstimator):
    # TODO: add returning of fragments
    # TODO: add morgan fingerprint
    # TODO: return fragments without hashing (dict = frag: count)
    def __init__(self, radius: int = 4, length: int = 1024,
                 number_active_bits: int = 2, number_bit_pairs: int = 4):
        """
        Utility tool for making Morgan-like fingerprints

        :param radius: maximum length of fragments
        :param length: bit string's length
        :param number_active_bits: number of active bits for each hashed tuple
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number, we will be
               activate only this number of fragments
        """
        self._radius = radius
        self._mask = length - 1
        self._length = length
        self._log = int(log2(length))
        self._number_active_bits = number_active_bits
        self._number_bit_pairs = number_bit_pairs

    def transform(self, x: Collection):
        bits = self.transform_bitset(x)
        fingerprints = zeros((len(x), self._length))

        for idx, lst in enumerate(bits):
            fingerprints[idx, lst] = 1

        return fingerprints

    def transform_bitset(self, x: Collection) -> List[List[int]]:
        all_active_bits, new_arr, hashes = [], [], set()
        for mol in x:
            arr = self._fragments(self._bfs(mol), mol)
            hashes = {tuple_hash((*tpl, cnt))
                      for tpl, count in arr.items()
                      for cnt in range(1, min(count, self._number_bit_pairs) + 1)}

            active_bits = set()
            for tpl in hashes:
                bit = tpl & self._mask - 1
                active_bits.add(bit)
                if self._number_active_bits == 2:
                    bit = tpl >> self._log & self._mask - 1
                    active_bits.add(bit)
                elif self._number_active_bits > 2:
                    for idx in range(1, self._number_active_bits):
                        bit = tpl >> self._log * idx & self._mask - 1
                        active_bits.add(bit)
            all_active_bits.append(list(active_bits))

        return all_active_bits

    def _bfs(self, molecule: "MoleculeContainer") -> List[List[int]]:
        atoms = molecule._atoms
        bonds = molecule._bonds

        arr = [[x] for x in atoms]
        queue = deque(arr)
        storage = defaultdict(int)
        while queue:
            now = queue.popleft()
            var = [now + [x] for x in bonds[now[-1]] if x not in now]

            # temporary solution about bug with doubling numbers of fragments
            for frag in var:
                canon_frag = tuple(frag)
                rev_canon_frag = canon_frag[::-1]
                if canon_frag > rev_canon_frag and not storage[canon_frag]:
                    storage[canon_frag] += 1
                    arr.append(frag)
                elif canon_frag < rev_canon_frag and not storage[rev_canon_frag]:
                    storage[rev_canon_frag] += 1
                    arr.append(frag)
                elif canon_frag == rev_canon_frag:
                    storage[canon_frag] += 1
                    arr.append(frag)

            if not var or len(var[0]) >= self._radius:
                continue
            queue.extend(var)
        return arr

    def _fragments(self, arr: List[List], molecule: "MoleculeContainer") -> Dict[Tuple[int, ...], int]:
        atoms = {x: int(a) for x, a in molecule.atoms()}
        bonds = molecule._bonds
        cache = defaultdict(dict)
        out = defaultdict(int)

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
                out[var] += 1
            else:
                out[rev_var] += 1
        return out


__all__ = ['MorganFingerprint']
