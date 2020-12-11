


import numpy as np

class LUT ():
    def __init__ (self, n=2, is3D=False, color_step=20, rng=None, initial=None, alpha_only=None):
        if rng is None:
            rng = np.random
        self.rng = rng
        self.is3D = is3D
        if not self.is3D:
            self.table = np.zeros ((256, 3), dtype=np.int32)
            for i in range (256):
                self.table [i] = np.array ([i, i, i])
        else:
            if initial is None:
                self.table = np.zeros ((256, 4), dtype=np.int32)
                for i in range (256):
                    self.table [i] = np.array ([i, i, i, i])
            else:
                self.table = np.zeros ((256, 4), dtype=np.int32)
                for i in range (256):
                    self.table [i] = np.array ([i, i, i, i])
                step = 256 // (len (initial)-1)
                l = 0
                for i in range (len (initial)):
                    r = step * i
                    r = self.clip (r)
                    self.table [r][3] = initial [i]
                    self.linear_adjust_r (l, r, 3)
                    l = r
                self.linear_adjust_r (l, 255, 3)
        
        self.n = n
        self.step = 256 // (self.n - 1)
        self.mod = np.zeros ((self.n), dtype=np.int32)
        self.color_step=color_step
        self.mod_rate = 0.6
        self.alpha_only = alpha_only

    def apply (self, img):
        ret = self.table [img]
        return ret

    def rand_mod (self):
        rng = self.rng
        n = self.n
        step = self.step
        for c in range (self.table.shape[-1]):
            l = 0
            for i in range (n):
                
                if self.rng.rand () > self.mod_rate:
                    continue

                r = self.clip (i * step)
                
                mod = 0
                if self.alpha_only:
                    # Limited mod
                    mod = self.rng.choice  (list (range (-4, 5)), 1) [0] * self.color_step
                    self.table [r][c] += mod

                ## Fixed mod
                # mod = self.rng.choice  ([-1 * 3, 0, 1 * 3], 1) [0] * self.color_step
                # self.table [r][c] += mod             
                
                if not self.alpha_only:
                    # Full mod
                    mod = self.rng.randint (-128, 128)
                    self.table [r][c] += mod 
                    self.table [r][c] = self.rng.choice  (list (range (0, 256)), 1) [0]  
                
                self.table [r][c] = self.clip (self.table [r][c])
                self.mod [i] = mod
                self.linear_adjust_r (l, r, c)
                l = r
            # l = self.clip ((n - 1) * step); r = 255
            r = 255
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
        mid = self.clip (i * step)
        l = self.clip ((i - 1) * step)
        r = self.clip ((i + 1) * step)
        self.table [mid][c] += amount
        self.table [mid][c] = self.clip (self.table[mid][c])
        self.linear_adjust_r (l, mid, c)
        self.linear_adjust_r (mid, r, c)

    def cmp (self, other, i, c):
        return (abs (self.table[self.clip (i * self.step)][c] - other.table[self.clip (i * self.step)][c]))
