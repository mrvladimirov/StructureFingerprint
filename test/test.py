from CGRtools.files import SDFRead
from MorganFingerprint import bfs_pure_python, bfs_numba, bfs_numba_parallel, convert_to_matrix
from time import time


with SDFRead('25.sdf') as f:
    mols = f.read()
mol = mols[8]
arr, adj = convert_to_matrix(mol)
now = time()
arr1 = bfs_pure_python(mol, 6)
time_1 = time() - now
now = time()
arr2 = bfs_numba(arr, adj, 6)
time_2 = time() - now
now = time()
arr3 = bfs_numba(arr, adj, 6)
time_3 = time() - now
print(f'Pure python: {time_1}, numba: {time_2}, numba repeat: {time_3}')
