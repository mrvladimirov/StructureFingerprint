# -*- coding: utf-8 -*-
#
#  Copyright 2021 Aleksandr Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from importlib.util import find_spec
from math import log2
from numpy import zeros, uint8
from pkg_resources import get_distribution
from typing import Collection, List, Union


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


class MorganFingerprint(TransformerMixin, BaseEstimator):
    def __init__(self, min_radius: int = 1, max_radius: int = 4, length: int = 1024, number_active_bits: int = 2,
                 include_hydrogens: bool = True):
        """
        Morgan fingerprints. Similar to RDkit implementation.

        :param min_radius: minimal radius of EC
        :param max_radius: maximum radius of EC
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        :param include_hydrogens: take into account hydrogen atoms
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.length = length
        self.number_active_bits = number_active_bits
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
        for mol in x:
            active_bits = set()
            for tpl in self._morgan(mol):
                active_bits.add(tpl & mask)
                if number_active_bits == 2:
                    active_bits.add(tpl >> log & mask)
                elif number_active_bits > 2:
                    for _ in range(1, number_active_bits):
                        tpl >>= log
                        active_bits.add(tpl & mask)
            all_active_bits.append(list(active_bits))
        return all_active_bits

    def transform_hashes(self, x: Collection[Union[MoleculeContainer, CGRContainer]]) -> List[List[int]]:
        """
        Transform structures into list of integer hashes of atoms with EC.

        :param x: CGRtools MoleculeContainer or CGRContainer
        :return: list of list of integer hashes
        """
        return [self._morgan(mol) for mol in x]

    def _morgan(self, molecule: Union[MoleculeContainer, CGRContainer]) -> List[int]:
        min_radius = self.min_radius

        if not isinstance(molecule, (MoleculeContainer, CGRContainer)):
            raise TypeError('MoleculeContainer or CGRContainer expected')

        if self.include_hydrogens and isinstance(molecule, MoleculeContainer):
            identifiers = {idx: tuple_hash((atom, atom.implicit_hydrogens)) for idx, atom in molecule.atoms()}
        else:
            identifiers = {idx: int(atom) for idx, atom in molecule.atoms()}

        bonds = molecule._bonds
        arr = set()
        for step in range(1, self.max_radius + 1):
            if step >= min_radius:
                arr.update(identifiers.values())
            identifiers = {idx: tuple_hash((tpl, *(x for x in
                                                   sorted((int(b), identifiers[ngb]) for ngb, b in bonds[idx].items())
                                                   for x in x)))
                           for idx, tpl in identifiers.items()}

        if self.max_radius > 1:  # add last ring
            arr.update(identifiers.values())
        return list(arr)


__all__ = ['MorganFingerprint']
