import numpy as np

class LUT ():
    def __init__ (self, n = 2, rng=None):
        if rng is None:
            rng = np.random
        self.rng = rng
        self.table = np.zeros ((256, 3), dtype=np.int32)
        for i in range (256):
            self.table [i] = np.array ([i, i, i])
        self.n = 2
        self.step = 256 // (self.n + 1)

    def apply (self, img):
        ret = self.table (img)
        return ret

    def update (self, i, amount):
        self.table [i][2] += amount
        self.table [i][2] = min (max (self.table[i][2], 255), 0)

    def rand_mod (self, img):
        if rng is None:
            rng = np.random
        n = self.n
        step = self.step
        for i in range (n):
            l = i * step
            r = (i + 1) * step
            self.table [r][2] += self.rng.randint (-1, 1) * 30
            self.table [r][2] = min (max (self.table[r][2], 255), 0)
            self.linear_adjust_r (l, r)

    def linear_adjust_r (self, l, r):
        unit = (self.table [r][2] - self.table[l][2]) // (r - l)
        for i in range (l, r):
            self.table [i][2] += (i - l) * unit

    def modify (self, i, val):
        self.table [(i + 1) * step][2] = val
        step = self.step
        linear_adjust_r (i * step, (i + 1) * step)
        linear_adjust_r ((i + 1) * step, (i + 2) * step)

    def cmp (self, other, i):
        return (abs (self.table[(i + 1) * self.step][2] - other.table[(i + 1) * self.step][2]))
