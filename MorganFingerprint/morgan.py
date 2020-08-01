from numba import njit
from numpy import append, array, empty, ndenumerate, ones, uint64, zeros


class MorganFingerprint:
    @staticmethod
    def convert_to_matrix(molecule):
        atoms = array([v for k, v in molecule.atoms()], dtype=uint64)
        adj = zeros((len(atoms) + 1, len(atoms) + 1), dtype=uint64)
        for k, v, b in molecule.bonds():
            adj[k][v] = adj[v][k] = b.order
        return atoms, adj

    @staticmethod
    @njit(parallel=True)
    def chains(atoms, adj, length):
        max_number = len(atoms) * 4 * 3**(length - 1)
        fragments = zeros((max_number, len(atoms)), dtype=uint64)
        index = 0
        for i, x in ndenumerate(atoms):
            stack = array([(i[0] + 1,), (0,)], dtype=uint64)
            path = [uint64(x) for x in range(0)]
            while stack.size > 0:
                not_seen = ones(len(atoms) + 1, dtype=uint64)
                for was in path:
                    not_seen[was] = 0
                now, depth = stack[0][-1], stack[1][-1]
                old_stack = stack
                stack = zeros((len(stack), len(stack[0]) - 1), dtype=uint64)
                for k in range(len(old_stack)):
                    stack[k] = old_stack[k][:-1]
                if len(path) > depth:
                    path = path[:depth]
                path.append(now)
                if len(path) <= length:
                    print(path)
                    fragments[index][:len(path)] = array(path, dtype=uint64)
                    index = index + 1
                depth += 1
                if depth + 1 > length:
                    continue
                for z, b in ndenumerate(adj[now]):
                    if not_seen[z[0]] and b:
                        stack = append(stack, array([(z[0],), (depth,)], dtype=uint64), axis=1)
        return fragments

    @staticmethod
    def tuple_hash(v):
        """
        Python 3.8 hash for tuples implemented on python.
        Working only for nonnested tuples.
        """
        acc = 0x27D4EB2F165667C5
        for el in v:
            acc += el * 0xC2B2AE3D27D4EB4F
            acc = (acc << 31) | (acc >> 33)
            acc *= 0x9E3779B185EBCA87

        acc += len(v) ^ 2870177450013471926  # 0xC2B2AE3D27D4EB4F ^ 3527539

        if acc == 0xFFFFFFFFFFFFFFFF:
            return 1546275796
        return acc


__all__ = ['MorganFingerprint']
