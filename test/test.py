from CGRtools.files import SDFRead
from MorganFingerprint import bfs_pure_python, bfs_numba, dfs, convert_to_matrix, convert_to_another
from time import time

sorted
with SDFRead('2500.sdf') as f:
    mols = f.read()
pure_all, numba_all, dfs_all = [], [], []
for i, mol in enumerate(reversed(mols), start=1):
    now = time()
    atoms, matrix = convert_to_another(mol)
    arr = bfs_numba(atoms, matrix, 4)
    numba_all.append(time() - now)
    now = time()
    arr = bfs_pure_python(mol, 4)
    pure_all.append(time() - now)
    now = time()
    atoms, matrix = convert_to_matrix(mol)
    arr = dfs(atoms, matrix, 4)
    dfs_all.append(time() - now)
    print(i)
pure = sum(pure_all)
numba = sum(numba_all)
dfs_time = sum(dfs_all)
print(f'Summary time of pure python is {pure}, numba summary time is {numba}, dfs sumtime is {dfs_time}'
      f'Average time for pure python is {pure/len(pure_all)}, average time for numba is {numba/len(numba_all)},'
      f'average time for dfs is {dfs_time/len(dfs_all)}')
