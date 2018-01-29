import warnings
from copy import deepcopy

import numpy as np
import itertools
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from sklearn.preprocessing import scale
from hmmlearn import hmm

# hyper parameters
ZERO = 1.e-10
ONE = 1
INF = 1.e+10
FB = 4 * 8
INFER_ITER_MAX = 10
NSAMPLE = 10
REGIME_R = 0.03
LM = 0.1
# for HMM
MINK = 1
MAXK = 8
N_INFER_ITER_HMM = 5
MAXBAUMN = 3

global ws
global X
global N
global dim

warnings.filterwarnings('ignore')

class Segbox():
    def __init__(self):
        self.subs = []
        self.len = 0
        self.costT = self.costC = INF
        self.model = None
        self.delta = 1

    def add_segment(self, st, ln):
        # print('add segment:', st, ln)
        if ln <= 0: return
        if st < 0: st = 0
        loc = 0
        for i in range(len(self.subs)):
            loc += 1
            if self.subs[i][0] > st: break
        self.subs.insert(loc, [st, ln])
        # remove overlap
        curr = INF
        while curr > len(self.subs):
            curr = len(self.subs)
            for i in range(curr - 1):
                st0, ln0 = self.subs[i]
                st1, ln1 = self.subs[i + 1]
                ed0, ed1 = st0 + ln0, st1 + ln1
                ed = ed0 if ed0 > ed1 else ed1
                if ed0 + 1 >= st1:
                    self.subs.pop(i + 1)
                    self.subs[i][1] = ed - st0
        # print('remove overlap:', self.subs)
        _, self.len = np.sum(np.array(self.subs), axis=0)

    def add_segment_ex(self, st, ln):
        self.len += ln
        self.subs.append((st, ln)) #!!!

class PlaitWS():
    def __init__(self, X):
        # self.input = X
        self.n, self.d = X.shape
        self.C = []
        self.Opt = []
        self.costT = 0.

def subsequences(s):
    subs = s.subs
    X_list, lengths = [], []
    for i in range(len(subs)):
        st, ln = subs[i]
        X_list.append(X[st:st + ln])
        lengths.append(ln)
    # print(X_list)
    # print(lengths)
    return X_list, lengths

def log_s(x):
    return 2. * np.log2(x) + 1.

def GaussianPDF(mean, var, x):
    var = np.fabs(var)
    p = np.exp(-(x - mean) * (x - mean) / (2 * var)) / np.sqrt(2 * np.pi * var)
    if p >= 1.: p = ONE
    if p <= 0.: p = ZERO
    # print(p)
    return p

def pdfL(hmm, kid, x):
    p = 0.
    mean = hmm.means_[kid]
    var = np.max(hmm.covars_[kid], axis=0)
    for i in range(dim):
        p += np.log(ZERO + GaussianPDF(mean[i], var[i], x[i]))
    if p < np.log(ZERO):
        p = np.log(ZERO)
    return p

def costHMM(k, d):
    return FB * (k + k*k + 2*k*d) + 2.0 * np.log(k) / np.log(2.0) + 1.0

def MDLsegment(stack):
    return sum([s.costT for s in stack])

def _viterbi(hmm, delta, st, ln):
    Lh = hmm.score(X[st:st + ln])
    if delta <= 0 or delta >= 1:
        exit('not appropriate delta')
    Lh += np.log(delta)
    Lh += np.log(1 - delta)
    costC = -Lh / np.log(2.)
    return costC

def computeLhMDL(s):
    if not len(s.subs):
        s.costC = s.costT = INF; return
    s.costC = 0.
    for i in range(len(s.subs)):
        st, ln = s.subs[i]
        s.costC += _viterbi(s.model, s.delta, st, ln)
    s.costT = _MDL(s)

def _MDL(s):
    m = len(s.subs)
    k = s.model.n_components
    costT = costLen = 0.
    costC = s.costC
    costM = costHMM(k, dim)
    for i in range(m):
        costLen += np.log2(s.subs[i][1])
    costLen += m * np.log2(k)
    return costC + costLen + costM # i.e., costT

def _MDLtotal():
    r = len(ws.Opt) + len(ws.C)
    m = sum([len(sbox.subs) for sbox in ws.Opt])
    m += sum([len(sbox.subs) for sbox in ws.C])
    cost = MDLsegment(ws.Opt) + MDLsegment(ws.C)
    costT = cost + log_s(r) + log_s(m) + m*np.log2(r) + FB*r*r
    print(r, m, costT)
    return costT

def estimateHMM_k(s, k=1):
    X_tmp, lengths = subsequences(s)
    if len(X_tmp) > MAXBAUMN:
        X_tmp = X_tmp[:MAXBAUMN]
        lengths = lengths[:MAXBAUMN]
    X_flat = np.concatenate(X_tmp)
    s.model = hmm.GaussianHMM(
        n_components=k,
        covariance_type='diag', # full, diag
        n_iter=N_INFER_ITER_HMM
    )
    s.model.fit(X_flat, lengths=lengths)
    # print('# of states:', s.model.n_components)
    # print('init prob.:\n', s.model.startprob_)
    # print('trans prob.:\n', s.model.transmat_)
    # print('means:\n', s.model.means_)
    # print('covariance:\n', s.model.covars_)
    s.delta = len(s.subs) / s.len

def estimateHMM(s):
    # print('estimate HMM...')
    s.costT = INF
    optk = MINK
    for k in range(MINK, MAXK):
        prev = s.costT
        estimateHMM_k(s, k)
        computeLhMDL(s)
        # print(s.costT)
        if s.costT > prev:
            optk = k - 1
            break
    if optk < MINK: optk = MINK
    if optk > MAXK: optk = MAXK
    # print('opt-k:', optk)
    estimateHMM_k(s, optk)
    computeLhMDL(s)

def search_aux(st, length, s0, s1):
    d0, d1 = s0.delta, s1.delta
    if d0 <= 0 or d1 <= 0: error('delta is zero.')
    m0, m1 = s0.model, s1.model
    k0, k1 = m0.n_components, m1.n_components
    Pu, Pv = np.zeros(k0), np.zeros(k0)
    Pi, Pj = np.zeros(k1), np.zeros(k1)
    Su, Sv = [[] for _ in range(k0)], [[] for _ in range(k0)]
    Si, Sj = [[] for _ in range(k1)], [[] for _ in range(k1)]
    # t = 0
    t = st
    for v in range(k0):
        Pv[v] = np.log(d1)
        Pv[v] += np.log(m0.startprob_[v] + ZERO)
        Pv[v] += pdfL(m0, v, X[t])
    for j in range(k1):
        Pj[j] = np.log(d0)
        Pj[j] += np.log(m1.startprob_[j] + ZERO)
        Pj[j] += pdfL(m1, j, X[t])
    # t >= 1
    for t in range(st + 1, st + length):
        # Pu(t)
        maxj = np.argmax(Pj)
        for u in range(k0):
            maxPj = Pj[maxj] + np.log(d1)
            maxPj += np.log(m0.startprob_[u] + ZERO)
            maxPj += pdfL(m0, u, X[t])
            maxPv, maxv = -INF, -1
            for v in range(k0):
                val = np.log(1.0 - d0) + Pv[v]
                val += np.log(m0.transmat_[v][u] + ZERO)
                val += pdfL(m0, u, X[t])
                if val > maxPv: maxPv, maxv = val, v
            if maxPj > maxPv:
                Pu[u] = maxPj
                Su[u] = deepcopy(Sj[maxj])
                Su[u].append(t)
            else:
                Pu[u] = maxPv
                Su[u] = deepcopy(Sv[maxv])
        # Pj(t)
        maxv = np.argmax(Pv)
        for i in range(k1):
            maxPv = Pv[maxv] + np.log(d0)
            maxPv += np.log(m1.startprob_[i] + ZERO)
            maxPv += pdfL(m1, i, X[t])
            maxPj, maxj = -INF, -1
            for j in range(k1):
                val = np.log(1.0 - d1) + Pj[j]
                val += np.log(m1.transmat_[j][i] + ZERO)
                val += pdfL(m1, i, X[t])
                if val > maxPj: maxPj, maxj = val, j
            if maxPv > maxPj:
                Pu[u] = maxPj
                Su[u] = deepcopy(Sj[maxj])
                Su[u].append(t)
            else:
                Pu[u] = maxPv
                Su[u] = deepcopy(Sv[maxv])
        tmp = deepcopy(Pu); Pu = deepcopy(Pv); Pv = deepcopy(tmp)
        tmp = deepcopy(Pi); Pi = deepcopy(Pj); Pj = deepcopy(tmp)
        tmp = deepcopy(Su); Su = deepcopy(Sv); Sv = deepcopy(tmp)
        tmp = deepcopy(Si); Si = deepcopy(Sj); Sj = deepcopy(tmp)
    # end for
    maxv = np.argmax(Pv)
    maxj = np.argmax(Pj)
    if Pv[maxv] > Pj[maxj]:
        path = Sv[maxv]
        firstID = pow(-1, len(path)) * 1
        lh = Pv[maxv]
    else:
        path = Sj[maxj]
        firstID = pow(-1, len(path)) * -1
        lh = Pj[maxj]
    # add paths
    curst = st
    for i in range(len(path)):
        nxtst = path[i]
        if firstID * pow(-1, i) == 1:
            s0.add_segment(curst, nxtst - curst)
        else:
            s1.add_segment(curst, nxtst - curst)
        curst = nxtst
    if firstID * pow(-1, len(path)) == 1:
        s0.add_segment(curst, st + length - curst)
    else:
        s1.add_segment(curst, st + length - curst)
    return -lh / np.log(2.) # i.e., costC

def cut_point_search(Sx, s0, s1, RM=True):
    s0.subs.clear()
    s1.subs.clear()
    lh = 0.
    for i in range(len(Sx.subs)):
        lh += search_aux(Sx.subs[i][0], Sx.subs[i][1], s0, s1)
    # if RM: remove_noise(Sx, s0, s1)
    return lh

def copy_segments(s0, s1): # from s0 to s1
    s1.subs = deepcopy(s0.subs)

def select_largest(s):
    loc = np.argmax(np.array(s.subs), axis=0)[1] 
    st, ln = s.subs[loc]
    s.subs.clear()
    s.add_segment(st, ln)

def uniformset(Sx, length, trial):
    u = Segbox()
    slide_w = int((Sx.len - length) / trial)
    for i in range(len(Sx.subs)):
        if len(u.subs) >= trial:
            return u
        st, ln = Sx.subs[i]
        ed = st + ln
        for j in range(trial):
            nxt = st + j * slide_w
            if nxt + length > ed:
                st = ed - length
                if st < 0: st = 0
                u.add_segment_ex(st, length)
                break
            u.add_segment_ex(nxt, length)
    return u

def fixed_sampling(Sx, length):
    s0, s1 = Segbox(), Segbox()
    loc = 0 % len(Sx.subs)
    r = Sx.subs[loc][0]
    s0.add_segment(r, length)
    loc = 1 % len(Sx.subs)
    r = Sx.subs[loc][0] + int(Sx.subs[loc][1] / 2)
    s1.add_segment(r, length)
    return s0, s1

def uniform_sampling(Sx, length, n1, n2, u):
    s0, s1 = Segbox(), Segbox()
    i, j = int(n1 % len(u.subs)), int(n2 % len(u.subs))
    st0, st1 = u.subs[i][0], u.subs[j][0]
    if abs(st0 - st1) < length: return s0, s1
    s0.add_segment(st0, length)
    s1.add_segment(st1, length)
    return s0, s1

def _find_centroid(Sx, n_samples, seedlen):
    u = uniformset(Sx, seedlen, n_samples)
    nsub = len(u.subs)
    costMin = INF
    for iter1, iter2 in itertools.product(range(nsub), repeat=2):
        s0, s1 = uniform_sampling(Sx, seedlen, iter1, iter2, u)
        if not len(s0.subs) or not len(s1.subs): continue
        s0stC, s0lenC = s0.subs[0]
        s1stC, s1lenC = s1.subs[0]
        estimateHMM_k(s0, MINK)
        estimateHMM_k(s1, MINK)
        cut_point_search(Sx, s0, s1)
        computeLhMDL(s0); computeLhMDL(s1)
        if not len(s0.subs) or not len(s1.subs): continue
        if costMin > s0.costT + s1.costT:
            costMin = s0.costT + s1.costT
            s0stB, s0lenB = s0stC, s0lenC
            s1stB, s1lenB = s1stC, s1lenC
    if costMin == INF:
        s0, s1 = fixed_sampling(Sx, seedlen)
        return s0, s1
    del s0, s1
    s0, s1 = Segbox(), Segbox()
    s0.add_segment(s0stB, s0lenB)
    s1.add_segment(s1stB, s1lenB)
    return s0, s1

def regimge_split(Sx):
    print('RegimeSplit')
    seedlen = int(N * LM)
    s0, s1 = _find_centroid(Sx, NSAMPLE, seedlen)
    print(s0.subs, '\n', s1.subs)
    opt0, opt1 = Segbox(), Segbox()
    for i in range(INFER_ITER_MAX):
        # select largest
        select_largest(s0)
        select_largest(s1)
        # estimate HMM
        estimateHMM(s0); estimateHMM(s1)
        # cut point search
        cut_point_search(Sx, s0, s1)
        print(s0.subs, '\n', s1.subs)
        computeLhMDL(s0); computeLhMDL(s1)
        if not len(s0.subs) or not len(s1.subs): break
        diff = opt0.costT + opt1.costT
        diff -= s0.costT + s1.costT
        if diff > 0:
            copy_segments(s0, opt0)
            copy_segments(s1, opt1)
        elif i >= INFER_ITER_MIN: break
    copy_segments(opt0, s0)
    copy_segments(opt1, s1)
    del opt0, opt1
    if not len(s0.subs) or not len(s1.subs):
        return s0, s1
    estimateHMM(s0); estimateHMM(s1)
    return s0, s1


def autoplait(X):
    # set initial segment
    Sx = Segbox()
    Sx.add_segment(0, len(X))
    estimateHMM(Sx)
    ws.C.append(Sx)

    # main loop
    while True:
        costT = _MDLtotal()
        Sx = ws.C.pop()
        # try to split regime: s0, s1
        s0, s1 = regimge_split(Sx)
        costT_s01 = s0.costT + s1.costT
        print(costT_s01 + Sx.costT*REGIME_R, 'vs', Sx.costT)
        if costT_s01 + Sx.costT*REGIME_R < Sx.costT:
            ws.C.append(s0)
            ws.C.append(s1)
        else:
            ws.Opt.append(Sx)
        if not ws.C: break


if __name__ == '__main__':

    X = np.loadtxt('./dat/21_01.amc.4d')
    X = np.loadtxt('./dat/86_01.amc.4d')
    X = scale(X)
    N, dim  = X.shape
    ws = PlaitWS(X)

    print('-----------------')
    print('| r | m | costT |')
    print('-----------------')
    result = autoplait(X)

    plt.subplot(211)
    plt.plot(X)
    plt.subplot(212)
    plt.show()