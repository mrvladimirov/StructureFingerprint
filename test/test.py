from CGRtools.files import SDFRead
from CIMtools.preprocessing import FragmentorFingerprint
from MorganFingerprint import MorganFingerprint
from time import time

with SDFRead('2500.sdf') as f:
    mols = f.read()

morgan = MorganFingerprint()
fragmentor = FragmentorFingerprint(fingerprint_size=10, min_length=1, max_length=4, doallways=True)

now = time()
fingers = morgan.transform(mols)
print(f'My time is {time() - now}')
now = time()
fingers = fragmentor.transform(mols)
print(f'Their time is {time() - now}')
