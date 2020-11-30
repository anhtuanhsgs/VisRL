import numpy as np

class LUT ():
    def __init__ (self, n=2, rng=None):
        if rng is None:
            rng = np.random
        self.rng = rng
        self.table = np.zeros ((256, 3), dtype=np.int32)
        for i in range (256):
            self.table [i] = np.array ([i, i, i])
        self.n = 2
        self.step = 256 // (self.n + 1)
        self.mod = np.zeros ((self.n), dtype=np.int32)

    def apply (self, img):
        ret = self.table [img]
        return ret

    def rand_mod (self):
        rng = self.rng
        n = self.n
        step = self.step
        for i in range (n):
            l = i * step
            r = (i + 1) * step
            mod = self.rng.choice  ([-1, 1], 1) [0] * 60
            self.table [r][2] += mod
            self.table [r][2] = self.clip (self.table [r][2])
            self.mod [i] = mod
            self.linear_adjust_r (l, r)
        l = n * step; r = 255
        self.linear_adjust_r (l, r)

    def linear_adjust_r (self, l, r):
        unit = 1.0 * (self.table [r][2] - self.table[l][2]) / (r - l)
        print ("Unit", unit, "l", l, "r", r)
        for i in range (l, r):
            self.table [i][2] = int (self.table [l][2] +  (i - l) * unit)

    def clip (self, x, l=0, r=255):
        return max (min (255, x), 0)

    def modify (self, i, amount):
        self.table [(i + 1) * step][2] += amount
        l = self.clip (i * step); r = self.clip ((i + 1) * step)
        linear_adjust_r (i * step, (i + 1) * step)
        linear_adjust_r ((i + 1) * step, (i + 2) * step)

    def cmp (self, other, i):
        return (abs (self.table[(i + 1) * self.step][2] - other.table[(i + 1) * self.step][2]))
