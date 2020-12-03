


import numpy as np

class LUT ():
    def __init__ (self, n=2, color_step=20, rng=None):
        if rng is None:
            rng = np.random
        self.rng = rng
        self.table = np.zeros ((256, 3), dtype=np.int32)
        for i in range (256):
            self.table [i] = np.array ([i, i, i])
        self.n = n
        self.step = 256 // (self.n - 1)
        self.mod = np.zeros ((self.n), dtype=np.int32)
        self.color_step=color_step

    def apply (self, img):
        ret = self.table [img]
        return ret

    def rand_mod (self):
        rng = self.rng
        n = self.n
        step = self.step
        for c in range (3):
            for i in range (n):
                l = self.clip ((i - 1) * step)
                r = self.clip (i * step)
                mod = 0

                # Limited mod
                # mod = self.rng.choice  (list (range (-4, 5)), 1) [0] * self.color_step
                
                # Fixed mod
                # mod = self.rng.choice  ([-1 * 4, 1 * 4], 1) [0] * self.color_step
                
                # self.table [r][c] += mod
                
                # Full mod
                self.table [r][c] = self.rng.choice  (list (range (0, 256)), 1) [0]  
                self.table [r][c] = self.clip (self.table [r][c])
                self.mod [i] = mod
                self.linear_adjust_r (l, r, c)
            l = self.clip ((n - 1) * step); r = 255
            self.linear_adjust_r (l, r, c)

    def linear_adjust_r (self, l, r, c):
        if l < 0 or r > 255 or l >= r:
            return
        unit = 1.0 * (self.table [r][c] - self.table[l][c]) / (r - l)
        for i in range (l, r):
            self.table [i][c] = int (self.table [l][c] +  (i - l) * unit)

    def clip (self, x, l=0, r=255):
        return max (min (255, x), 0)

    def modify (self, i, c, amount):
        step = self.step
        self.table [i * step][c] += amount
        self.table [i * step][c] = self.clip (self.table[i * step][c])
        self.linear_adjust_r ((i - 1) * step, i * step, c)
        self.linear_adjust_r (i * step, (i + 1) * step, c)

    def cmp (self, other, i, c):
        return (abs (self.table[i * self.step][c] - other.table[i * self.step][c]))
