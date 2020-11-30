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

    def apply (self, img):
        ret = self.table [img]
        return ret

    def update (self, i, amount):
        self.table [i][2] += amount
        self.table [i][2] = self.clip (self.table [i][2])

    def rand_mod (self):
        rng = self.rng
        n = self.n
        step = self.step
        for i in range (n):
            l = i * step
            r = (i + 1) * step
            print ("left:, ", l, "right: ", r)
            self.table [r][2] += self.rng.randint (-1, 1) * 60
            self.table [r][2] = self.clip (self.table [r][2])
            self.linear_adjust_r (l, r)
        l = i * step, r = 255
        self.linear_adjust_r (l, r)

    def linear_adjust_r (self, l, r):
        unit = (self.table [r][2] - self.table[l][2]) // (r - l)
        for i in range (l, r):
            self.table [i][2] += (i - l) * unit

    def clip (self, x, l=0, r=255):
        return max (min (255, x), 0)

    def modify (self, i, val):
        self.table [(i + 1) * step][2] = val
        l = self.clip (i * step); r = self.clip ((i + 1) * step)
        linear_adjust_r (i * step, (i + 1) * step)
        linear_adjust_r ((i + 1) * step, (i + 2) * step)

    def cmp (self, other, i):
        return (abs (self.table[(i + 1) * self.step][2] - other.table[(i + 1) * self.step][2]))
