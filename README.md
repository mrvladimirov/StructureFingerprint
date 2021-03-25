Molecules and CGR's fingerprints
--------------------------------

Library provides transformers of molecules and CGR's into features.

* Pure python code
* Sklearn like API. Pipelines supported.
* Linear fingerprints
* Morgan fingerprints
* Linear features lists
* Morgan features lists

Install
-------

    pip install StructureFingerprint

Example
-------

    import numpy as np
    from CGRtools import smiles
    from StructureFingerprint import LinearFingerprint, MorganFingerprint

    lfp = LinearFingerprint()  # Sklearn-like transformer
    mol = smiles('CN(C)C=O')

    print(lfp.transform([mol]))  # array. Can be used in sklearn estimators
    print(lfp.transform_bitset([mol]))  # active bits indexes

    print(set(np.where(lfp.transform([mol])==1)[1]) == set(lfp.transform_bitset([mol])[0]))

    print(lfp.transform_hashes([mol]))  # hashes of fragments
