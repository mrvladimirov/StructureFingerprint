# -*- coding: utf-8 -*-
#
#  Copyright 2020, 2021 Aleksandr Sizov <murkyrussian@gmail.com>
#  Copyright 2020, 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of StructureFingerprint.
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
from collections import defaultdict, deque
from importlib.util import find_spec
from math import log2
from numpy import zeros
from pkg_resources import get_distribution
from typing import Collection, TYPE_CHECKING, List, Dict, Tuple, Set, Deque, Union


cgr_version = get_distribution('CGRtools').version
if cgr_version.startswith('4.0.'):  # 4.0 compatibility
    from CGRtools.algorithms.morgan import tuple_hash
else:
    from CGRtools._functions import tuple_hash

if find_spec('sklearn'):  # use sklearn classes if available
    from sklearn.base import BaseEstimator, TransformerMixin
else:
    class BaseEstimator:
        ...

    class TransformerMixin:
        ...

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer, CGRContainer


class LinearFingerprint(TransformerMixin, BaseEstimator):
    def __init__(self, min_radius: int = 1, max_radius: int = 4, length: int = 1024,
                 number_active_bits: int = 2, number_bit_pairs: int = 4):
        """
        Linear fragments fingerprints

        :param min_radius: minimal length of fragments
        :param max_radius: maximum length of fragments
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number, we will be
               activate only this number of fragments
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.length = length
        self.number_active_bits = number_active_bits
        self.number_bit_pairs = number_bit_pairs

    def transform(self, x: Collection[Union['MoleculeContainer', 'CGRContainer']]):
        bits = self.transform_bitset(x)
        fingerprints = zeros((len(x), self.length))

        for idx, lst in enumerate(bits):
            fingerprints[idx, list(lst)] = 1
        return fingerprints

    def transform_bitset(self, x: Collection[Union['MoleculeContainer', 'CGRContainer']]) -> List[List[int]]:
        number_bit_pairs = self.number_bit_pairs
        number_active_bits = self.number_active_bits
        mask = self.length - 1
        log = int(log2(self.length))

        all_active_bits = []
        for mol in x:
            arr = self._fragments(mol)
            hashes = {tuple_hash((*tpl, cnt))
                      for tpl, count in arr.items()
                      for cnt in range(min(count, number_bit_pairs))}

            active_bits = set()
            for tpl in hashes:
                active_bits.add(tpl & mask)
                if number_active_bits == 2:
                    active_bits.add(tpl >> log & mask)
                elif number_active_bits > 2:
                    for _ in range(1, number_active_bits):
                        tpl >>= log  # shift
                        active_bits.add(tpl & mask)

            all_active_bits.append(list(active_bits))
        return all_active_bits

    def _chains(self, molecule: Union['MoleculeContainer', 'CGRContainer']) -> Set[Tuple[int, ...]]:
        queue: Deque[Tuple[int, ...]]  # typing
        min_radius = self.min_radius
        max_radius = self.max_radius
        atoms = molecule._atoms
        bonds = molecule._bonds

        if min_radius == 1:
            arr = {(x,) for x in atoms}
            if max_radius == 1:  # special case
                queue = None
            else:
                queue = deque(arr)
        else:
            arr = set()
            queue = deque((x,) for x in atoms)

        while queue:
            now = queue.popleft()
            var = [now + (x,) for x in bonds[now[-1]] if x not in now]
            if var:
                if len(var[0]) < max_radius:
                    queue.extend(var)
                if len(var[0]) >= min_radius:
                    for frag in var:
                        rev = frag[::-1]
                        arr.add(frag if frag > rev else rev)
        return arr

    def _fragments(self, molecule: Union['MoleculeContainer', 'CGRContainer']) -> Dict[Tuple[int, ...], int]:
        atoms = {x: int(a) for x, a in molecule.atoms()}
        bonds = molecule._bonds
        out = defaultdict(int)

        for frag in self._chains(molecule):
            var = [atoms[frag[0]]]
            for x, y in zip(frag, frag[1:]):
                var.append(int(bonds[x][y]))
                var.append(atoms[y])
            var = tuple(var)
            rev_var = var[::-1]
            if var > rev_var:
                out[var] += 1
            else:
                out[rev_var] += 1
        return dict(out)


__all__ = ['LinearFingerprint']
