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
from importlib.util import find_spec
from math import log2
from numpy import zeros
from pkg_resources import get_distribution
from typing import Collection, TYPE_CHECKING, List, Set, Union

from CGRtools import MoleculeContainer, CGRContainer


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
    def __init__(self, min_radius: int = 1, max_radius: int = 4, length: int = 1024, number_active_bits: int = 2):
        """
        Morgan fingerprints

        :param min_radius: minimal radius of EC
        :param max_radius: maximum radius of EC
        :param length: bit string's length. Should be power of 2
        :param number_active_bits: number of active bits for each hashed tuple
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.length = length
        self.number_active_bits = number_active_bits

    def transform(self, x: Collection[Union['MoleculeContainer', 'CGRContainer']]):
        bits = self.transform_bitset(x)
        fingerprints = zeros((len(x), self.length))

        for idx, lst in enumerate(bits):
            fingerprints[idx, list(lst)] = 1
        return fingerprints

    def transform_bitset(self, x: Collection[Union['MoleculeContainer', 'CGRContainer']]) -> List[List[int]]:
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

    def _morgan(self, molecule: Union['MoleculeContainer', 'CGRContainer']) -> Set[int]:
        min_radius = self.min_radius

        if isinstance(molecule, MoleculeContainer):
            identifiers = {idx: tuple_hash((atom.atomic_number, atom.isotope or 0, atom.charge, atom.is_radical,
                                            atom.implicit_hydrogens))
                           for idx, atom in molecule.atoms()}
        elif isinstance(molecule, CGRContainer):
            identifiers = {idx: tuple_hash((atom.atomic_number, atom.isotope or 0, atom.charge, atom.is_radical,
                                            atom.p_charge, atom.p_is_radical))
                           for idx, atom in molecule.atoms()}
        else:
            raise TypeError('MoleculeContainer or CGRContainer expected')

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
        return arr


__all__ = ['MorganFingerprint']
