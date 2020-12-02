import unittest
from MorganFingerprint import MorganFingerprint
from CGRtools import smiles


class TestMorganFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        self.morgan = MorganFingerprint()
        self.molecule = smiles('CC1=CC=CC=C1')  # toluene

    def test_bfs_sum(self):
        self.assertEqual(len(self.morgan._bfs(self.molecule)), 30)

    def test_equal_bfs_fragments(self):
        self.assertEqual(len(self.morgan._bfs(self.molecule)),
                         sum(self.morgan._fragments(self.morgan._bfs(self.molecule), self.molecule).values()))


if __name__ == '__main__':
    unittest.main()
