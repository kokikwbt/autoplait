import numpy as np
from sklearn.preprocessing import scale

INFER_ITER_MAX = 10

class Segbox():
    def __init__(self):
        self.subs = []
        self.len = 0
        self.costT = self.costC = 0.
        # model
        # delta

    def add_segment(st, ed):
        pass


class PlaitWS():
    def __init__(self, X):
        self.n, self.d = X.shape
        self.C = []
        self.Opt = []
        self.costT = 0.

def cut_point_search():
    pass


def regimge_split():
    seedlen = 0.
    # find centroid
    for _ in range(INFER_ITER_MAX):
        # select largest
        # estimate HMM
        # cut point search
        pass
    pass


def autoplait(X):
    ws = PlaitWS(X)
    while True:
        costT = 0.
        if not ws.C: break
        # create new segsets

        # try to split regime: s0, s1
        regimge_split()
        costT_s01 = 0.
        if(costT_s01 < costT):
            # append s0
            # append s1
            pass
        else:
            # append Sx to Opt
            pass
    return ws

if __name__ == '__main__':
    X = np.loadtxt('./dat/21_01.amc.4d')
    result = autoplait(X)
    print(result.Opt)
