import unittest
from StructureFingerprint import LinearFingerprint
from CGRtools import smiles


class TestMorganFingerprint(unittest.TestCase):
    def setUp(self):
        self.morgan = LinearFingerprint()
        self.molecule = smiles('CC1=CC=CC=C1')  # toluene

    def test_chains(self):
        self.assertEqual(self.morgan._chains(self.molecule),
                         {(2,), (5, 6, 7, 2), (7, 6, 5, 4), (5,), (4, 3), (5, 4), (3, 2, 1), (4, 3, 2, 1), (4, 3, 2),
                          (6, 5), (6, 7, 2), (7, 2, 3), (6, 5, 4, 3), (4,), (1,), (7,), (6, 5, 4), (7, 2, 3, 4), (2, 1),
                          (6, 7, 2, 1), (7, 6), (5, 4, 3, 2), (5, 4, 3), (3, 2), (3,), (7, 6, 5), (6,), (7, 2),
                          (6, 7, 2, 3), (7, 2, 1)})

    def test_fragments(self):
        self.assertEqual(self.morgan._fragments(self.molecule),
                         {(200,): 7, (200, 1, 200, 2, 200, 1, 200): 4, (200, 2, 200, 1, 200, 2, 200): 3,
                          (200, 1, 200): 4, (200, 2, 200): 3, (200, 2, 200, 1, 200): 7,
                          (200, 2, 200, 1, 200, 1, 200): 1, (200, 1, 200, 1, 200): 1})

    def test_bitset(self):
        out = self.morgan.transform_bitset([self.molecule] * 2)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 47)

    def test_transform(self):
        out = self.morgan.transform([self.molecule] * 2)
        self.assertEqual(out.shape, (2, 1024))
        self.assertEqual(out.sum(), 94.)


if __name__ == '__main__':
    unittest.main()
