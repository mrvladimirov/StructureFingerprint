from CGRtools.files import SDFRead
from MorganFingerprint import MorganFingerprint
from pickle import load, dump
from time import time

with SDFRead('2500.sdf') as f:
    mols = f.read()

morgan = MorganFingerprint()
with open('fitted_fragmentor.pickle', 'rb') as f:
    fragmentor = load(f)

mine, not_mine = [], []
mine_all, not_mine_all = {}, {}
for i, mol in enumerate(mols, start=1):
    now = time()
    finger = morgan.transform(mol)
    mine_all[i] = finger
    mine.append(time() - now)
    now = time()
    frag = fragmentor.transform([mol]).values
    not_mine_all[i] = frag
    not_mine.append(time() - now)
    print(f'{i} targets is done')
print(f'My fingerprints time is {sum(mine)}, their fingerprints time is {sum(not_mine)},'
      f'my average time if {sum(mine)/len(mine)}, their average time is{sum(not_mine)/len(not_mine)}')
