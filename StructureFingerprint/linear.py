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
from CGRtools import MoleculeContainer, CGRContainer
from collections import defaultdict, deque
from importlib.util import find_spec
from math import log2
from numpy import zeros, uint8
from pkg_resources import get_distribution
from typing import Collection, List, Dict, Tuple, Set, Deque, Union


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


class LinearFingerprint(TransformerMixin, BaseEstimator):
    def __init__(self, min_radius: int = 1, max_radius: int = 4, length: int = 1024,
                 number_active_bits: int = 2, number_bit_pairs: int = 4, include_hydrogens: bool = True):
        """
        Linear fragments fingerprints.
        Transform molecule or CGR structures into fingerprints based on linear fragments descriptors.
        Also count of fragments takes into account by activating multiple bits, but less or equal to `number_bit_pairs`.

        For example `CC` fragment found 4 times and `number_bit_pairs` is set to 3.
        In this case will be activated 3 bits: for count 1, for count 2 and for count 3.
        This gives intersection in bits with another structure with only 2 `CC` fragments.

        :param min_radius: minimal length of fragments
        :param max_radius: maximum length of fragments
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        :param number_bit_pairs: describe how much repeating fragments we can count in hashable
               fingerprint (if number of fragment in molecule greater or equal this number, we will be
               activate only this number of fragments
        :param include_hydrogens: take into account hydrogen atoms
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.length = length
        self.number_active_bits = number_active_bits
        self.number_bit_pairs = number_bit_pairs
        self.include_hydrogens = include_hydrogens

    def fit(self, x, y=None):
        return self

    def transform(self, x: Collection[Union[MoleculeContainer, CGRContainer]]):
        """
        Transform structures into array of binary features.

        :param x: CGRtools MoleculeContainer or CGRContainer
        :return: array(n_samples, n_features)
        """
        bits = self.transform_bitset(x)
        fingerprints = zeros((len(x), self.length), dtype=uint8)

        for idx, lst in enumerate(bits):
            fingerprints[idx, list(lst)] = 1
        return fingerprints

    def transform_bitset(self, x: Collection[Union[MoleculeContainer, CGRContainer]]) -> List[List[int]]:
        """
        Transform structures into list of indexes of True-valued features.

        :param x: CGRtools MoleculeContainer or CGRContainer
        :return: list of list of indexes
        """
        number_active_bits = self.number_active_bits
        mask = self.length - 1
        log = int(log2(self.length))

        all_active_bits = []
        for hashes in self.transform_hashes(x):
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

    def transform_hashes(self, x: Collection[Union[MoleculeContainer, CGRContainer]]) -> List[List[int]]:
        """
        Transform structures into list of integer hashes of fragments with count information.

        :param x: CGRtools MoleculeContainer or CGRContainer
        :return: list of list of integer hashes
        """
        number_bit_pairs = self.number_bit_pairs
        hashes = []
        for mol in x:
            hashes.append(list({tuple_hash((*tpl, cnt)) for tpl, count in self._fragments(mol).items()
                                for cnt in range(min(len(count), number_bit_pairs))}))
        return hashes

    def _chains(self, molecule: Union[MoleculeContainer, CGRContainer]) -> Set[Tuple[int, ...]]:
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

    def _fragments(self, molecule: Union[MoleculeContainer, CGRContainer]) -> Dict[Tuple[int, ...],
                                                                                   List[Tuple[int, ...]]]:
        if not isinstance(molecule, (MoleculeContainer, CGRContainer)):
            raise TypeError('MoleculeContainer or CGRContainer expected')

        if self.include_hydrogens and isinstance(molecule, MoleculeContainer):
            atoms = {idx: tuple_hash((atom, atom.implicit_hydrogens)) for idx, atom in molecule.atoms()}
        else:
            atoms = {idx: int(atom) for idx, atom in molecule.atoms()}

        bonds = molecule._bonds
        out = defaultdict(list)

        for frag in self._chains(molecule):
            var = [atoms[frag[0]]]
            for x, y in zip(frag, frag[1:]):
                var.append(int(bonds[x][y]))
                var.append(atoms[y])
            var = tuple(var)
            rev_var = var[::-1]
            if var > rev_var:
                out[var].append(frag)
            else:
                out[rev_var].append(frag)
        return dict(out)


__all__ = ['LinearFingerprint']
