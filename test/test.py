from CGRtools.files import SDFRead
from MorganFingerprint import MorganFingerprint


with SDFRead('25.sdf') as f:
    mols = f.read()
mol = mols[8]
morgan = MorganFingerprint()
atoms, adj = morgan.convert_to_matrix(mol)
frgs = morgan.chains(atoms, adj, 4)
print(frgs)
