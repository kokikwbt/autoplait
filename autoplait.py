import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pprint
from copy import deepcopy
from itertools import combinations
from warnings import filterwarnings
from sklearn.preprocessing import normalize, scale
from sklearn.mixture import log_multivariate_normal_density
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
from joblib import Parallel, delayed
filterwarnings('ignore')
pp = pprint.PrettyPrinter(indent=4)
cmap = cm.Set1
ZERO = 0 #1.e-10
INF = 1.e+10
MINK = 1
MAXK = 16
N_INFER_ITER_HMM = 1
INFER_ITER_MIN = 2
INFER_ITER_MAX = 10
MAXSEG = 100
SEGMENT_R = 3.e-2
REGIME_R = 3.e-2
MAXBAUMN = 10
FB = 8 * 8
RM = True
LM = .1
NSAMPLE = 10
parallel = False

class AutoPlait(object):
    def __init__(self):
        self.costT = np.inf
        self.regimes = []

    def solver(self, X):
        self.X = X
        self.n, self.d = _shape(X)
        reg = Regime()
        reg.add_segment(0, self.n)
        _estimate_hmm(X, reg)
        candidates = [reg]

        while True:
            self.costT = _mdl_total(self.regimes, candidates)
            reg = candidates.pop()

            # try to split regime: s0, s1
            reg0, reg1 = regime_split(X, reg)
            # print(reg0.subs[:reg0.n_seg], '0')
            # print(reg1.subs[:reg1.n_seg], '1')
            costT_s01 = reg0.costT + reg1.costT + REGIME_R * reg.costT
            print(f'\t-- try to split: {costT_s01:.6} vs {reg.costT:.6}')
            # print(s0.costT, s1.costT)

            if costT_s01 < reg.costT:
                candidates.append(reg0)
                candidates.append(reg1)
            else:
                self.regimes.append(reg)
            if not candidates:
                break

    def save(self):
        plt.subplot(211)
        plt.plot(self.X)
        plt.ylabel('Value')
        plt.subplot(212)
        for r in range(len(self.regimes)):
            print(self.regimes[r].subs[:self.regimes[r].n_seg], r)
            for i in range(self.regimes[r].n_seg):
                st, dt = self.regimes[r].subs[i]
                plt.plot([st, st + dt - 1], [r, r], color=cmap(r))
        plt.xlabel('Time')
        plt.ylabel('Regime ID')
        plt.tight_layout()
        plt.savefig('./result.png')
        plt.close()

def _mdl_total(stack0, stack1):
    r = len(stack0) + len(stack1)
    m = sum([regime.n_seg for regime in stack0])
    m += sum([regime.n_seg for regime in stack1])
    costT = MDLsegment(stack0) + MDLsegment(stack1)
    costT += log_s(r) + log_s(m) + m * np.log2(r) + FB * r ** 2
    # print(f'[r, m, total_cost] = {r}, {m}, {costT:.6}')
    print('====================')
    print(' r:\t', r)
    print(' m:\t', m)
    print(f' costT:\t{costT:.6}')
    print('====================')
    return costT

def regime_split(X, sx):
    opt0, opt1 = Regime(), Regime()
    n, d = _shape(X)
    seedlen = int(n * LM)
    s0, s1 = _find_centroid(X, sx, NSAMPLE, seedlen)
    if not s0.n_seg or not s1.n_seg:
        return opt0, opt1
    for i in tqdm(list(range(INFER_ITER_MAX)), desc='RegimeSplit'):
        select_largest(s0)
        select_largest(s1)
        _estimate_hmm(X, s0)
        _estimate_hmm(X, s1)
        cut_point_search(X, sx, s0, s1, RM=RM)
        if not s0.n_seg or not s1.n_seg:
            print("===> early optimized")
            break
        diff = (opt0.costT + opt1.costT) - (s0.costT + s1.costT)
        if diff > 0:
            copy_segments(s0, opt0)
            copy_segments(s1, opt1)
        elif i >= INFER_ITER_MIN:
            print("===> early optimized")
            break
    copy_segments(opt0, s0)
    copy_segments(opt1, s1)
    del opt0, opt1
    if not s0.n_seg or not s1.n_seg:
        return s0, s1
    _estimate_hmm(X, s0)
    _estimate_hmm(X, s1)
    return s0, s1

def _search_aux(X, st, dt, s0, s1):
    d0, d1 = s0.delta, s1.delta
    if d0 <= 0 or d1 <= 0: error('delta is zero.')
    m0, m1 = s0.model, s1.model
    k0, k1 = m0.n_components, m1.n_components
    Pu, Pv = np.zeros(k0), np.zeros(k0)  # log probability
    Pi, Pj = np.zeros(k1), np.zeros(k1)  # log probability
    Su, Sv = [[] for _ in range(k0)], [[] for _ in range(k0)]
    Si, Sj = [[] for _ in range(k1)], [[] for _ in range(k1)]

    t = st
    Pv = np.log(d1) + np.log(m0.startprob_ + ZERO)
    for v in range(k0):
        Pv[v] += gaussian_pdfl(X[t], m0.means_[v], m0.covars_[v])
    Pj = np.log(d0) + np.log(m1.startprob_ + ZERO)
    for j in range(k1):
        Pj[j] += gaussian_pdfl(X[t], m0.means_[v], m0.covars_[v])

    for t in range(st + 1, st + dt):
        # Pu(t)
        maxj = np.argmax(Pj)
        for u in range(k0):
            maxPj = Pj[maxj] + np.log(d1) + np.log(m0.startprob_[u] + ZERO) + gaussian_pdfl(X[t], m0.means_[u], m0.covars_[u])
            val = Pv + np.log(1. - d0) + np.log(m0.transmat_[:, u] + ZERO)
            for v in range(k0):
                val[v] += gaussian_pdfl(X[t], m0.means_[u], m0.covars_[u])
            maxPv, maxv = np.max(val), np.argmax(val)
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
            maxPv = Pv[maxv] + np.log(d0) + np.log(m1.startprob_[i] + ZERO) + gaussian_pdfl(X[t], m1.means_[i], m1.covars_[i])
            val = Pj + np.log(1. - d1) + np.log(m1.transmat_[:, i] + ZERO)
            for j in range(k1):
                val[j] += gaussian_pdfl(X[t], m1.means_[i], m1.covars_[i])
            maxPj, maxj = np.max(val), np.argmax(val)
            if maxPv > maxPj:
                Pi[i] = maxPv
                Si[i] = deepcopy(Sv[maxv])
                Si[i].append(t)
            else:
                Pi[i] = maxPj
                Si[i] = deepcopy(Sj[maxj])
        tmp = np.copy(Pu); Pu = np.copy(Pv); Pv = np.copy(tmp)
        tmp = np.copy(Pi); Pi = np.copy(Pj); Pj = np.copy(tmp)
        tmp = deepcopy(Su); Su = deepcopy(Sv); Sv = deepcopy(tmp)
        tmp = deepcopy(Si); Si = deepcopy(Sj); Sj = deepcopy(tmp)

    maxv = np.argmax(Pv)
    maxj = np.argmax(Pj)
    if Pv[maxv] > Pj[maxj]:
        path = Sv[maxv]
        firstID = pow(-1, len(path)) * 1
        llh = Pv[maxv]
    else:
        path = Sj[maxj]
        firstID = pow(-1, len(path)) * -1
        llh = Pj[maxj]

    curst = st
    for i in range(len(path)):
        nxtst = path[i]
        if firstID * pow(-1, i) == 1:
            s0.add_segment(curst, nxtst - curst)
        else:
            s1.add_segment(curst, nxtst - curst)
        curst = nxtst
    if firstID * pow(-1, len(path)) == 1:
        s0.add_segment(curst, st + dt - curst)
    else:
        s1.add_segment(curst, st + dt - curst)
    # print(path)
    # print('s0', s0.subs[:s0.n_seg])
    # print('s1', s1.subs[:s1.n_seg])
    return -llh / np.log(2.)  # data coding cost

def cut_point_search(X, sx, s0, s1, RM=True):
    s0.initialize()
    s1.initialize()
    lh = 0.
    for i in range(sx.n_seg):
        lh += _search_aux(X, sx.subs[i, 0], sx.subs[i, 1], s0, s1)
    if RM: remove_noise(X, sx, s0, s1)
    _compute_lh_mdl(X, s0)
    _compute_lh_mdl(X, s1)
    return lh

def _mdl(regime):
    m = regime.n_seg
    k = regime.model.n_components
    d = regime.model.n_features
    costT = costLen = 0.
    costC = regime.costC
    costM = costHMM(k, d)
    for i in range(m):
        costLen += np.log2(regime.subs[i, 1])
    costLen += m * np.log2(k)
    return costC + costM + costLen

def _viterbi(X, hmm, delta):
    if not 0 <= delta <= 1:
        exit('not appropriate delta')
    # print(hmm.startprob_)
    llh = hmm.score(X) + np.log(delta) + np.log(1 - delta)
    return -llh / np.log(2)  # data coding cost

def _compute_lh_mdl(X, regime):
    if regime.n_seg == 0:
        regime.costT = regime.costC = np.inf
        return
    regime.costC = 0.
    for i in range(regime.n_seg):
        st, dt = regime.subs[i]
        regime.costC += _viterbi(X[st:st+dt], regime.model, regime.delta)
    regime.costT = _mdl(regime)

def _shape(X):
    return X.shape if X.ndim > 1 else (len(X), 1)

def _parse_input(X, regime):
    n_seg = regime.n_seg
    if n_seg == 1:
        st, dt = regime.subs[0]
        return X[st:st+dt, :], [dt]
    n_seg = MAXBAUMN if n_seg > MAXBAUMN else n_seg
    subss = []
    lengths = []
    for st, dt in regime.subs[:n_seg]:
        subss.append(X[st:st+dt, :])
        lengths.append(dt)
    subss = np.concatenate(subss)
    return subss, lengths

def _estimate_hmm_k(X, regime, k=1):
    X_, lengths = _parse_input(X, regime)
    regime.model = GaussianHMM(n_components=k,
                               covariance_type='diag',
                               n_iter=N_INFER_ITER_HMM)
    regime.model.fit(X_, lengths=lengths)
    regime.delta = regime.n_seg / regime.len

def _estimate_hmm(X, regime):
    regime.costT = np.inf
    opt_k = MINK
    for k in range(MINK, MAXK):
        prev = regime.costT
        _estimate_hmm_k(X, regime, k)
        _compute_lh_mdl(X, regime)
        if regime.costT > prev:
            opt_k = k - 1
            break
    if opt_k < MINK: opt_k = MINK
    if opt_k > MAXK: opt_k = MAXK
    _estimate_hmm_k(X, regime, opt_k)
    _compute_lh_mdl(X, regime)

class Regime(object):
    def __init__(self):
        self.subs = np.zeros((MAXSEG, 2), dtype=np.int16)
        self.model = None
        self.delta = 1.
        self.initialize()

    def initialize(self):
        self.len = 0
        self.n_seg = 0
        self.costC = np.inf
        self.costT = np.inf

    def add_segment(self, st, dt):
        if dt <= 0: return
        st = 0 if st < 0 else st
        n_seg = self.n_seg
        if n_seg == MAXSEG:
            raise ValueError(" ")
        elif n_seg == 0:
            self.subs[0, :] = (st, dt)
            self.n_seg += 1
            self.len = dt
            self.delta = 1 / dt
        else:
            loc = 0
            while loc < n_seg:
                if st < self.subs[loc, 0]:
                    break
                loc += 1
            self.subs[loc+1:n_seg+1, :] = self.subs[loc:n_seg, :]
            self.subs[loc, :] = (st, dt)
            n_seg += 1
            # remove overlap
            curr = np.inf
            while curr > n_seg:
                curr = n_seg
                for i in range(curr - 1):
                    st0, dt0 = self.subs[i]
                    st1, dt1 = self.subs[i + 1]
                    ed0, ed1 = (st0 + dt0), (st1 + dt1)
                    ed = ed0 if ed0 > ed1 else ed1
                    if ed0 > st1:
                        # print('remove overlap !!', self.subs[:self.n_seg])
                        # time.sleep(5)
                        # self.subs.pop(i + 1)
                        self.subs[i+1:-1, :] = self.subs[i+2:, :]  # pop subs[i]
                        self.subs[i, 1] = ed - st0
                        n_seg -= 1
                        break
            self.n_seg = n_seg
            self.len = sum(self.subs[:n_seg, 1])
            self.delta = self.n_seg / self.len
            # print(self.subs[:self.n_seg])

    def add_segment_ex(self, st, dt):
        self.subs[self.n_seg, :] = (st, dt)
        self.n_seg += 1
        self.len += dt
        self.delta = self.n_seg / self.len

    def del_segment(self, loc):
        seg = self.subs[loc]
        self.subs[loc:-1, :] = self.subs[loc+1:, :]  # pop subs[i]
        self.n_seg -= 1
        self.len -= seg[1]
        self.delta = self.n_seg / self.len if self.len > 0 else ZERO
        return seg

def log_s(x):
    return 2. * np.log2(x) + 1.

def costHMM(k, d):
    return FB * (k + k*k + 2*k*d) + 2. * np.log(k) / np.log(2.) + 1.

def MDLsegment(stack):
    return np.sum([regime.costT for regime in stack])

def gaussian_pdfl(x, means, covars):
    n_dim = len(x)
    covars = np.diag(covars)
    lpr = -.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars))
                  + np.sum((means ** 2) / covars)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1. / covars).T))
    # print('lpr', lpr)
    return lpr

def find_mindiff(X, s0, s1):
    cost = np.inf
    loc = -1
    for i in range(s0.n_seg):
        st, dt = s0.subs[i]
        costC0 = _viterbi(X[st:st+dt], s0.model, s0.delta)
        costC1 = _viterbi(X[st:st+dt], s1.model, s1.delta)
        diff = abs(costC1 - costC0)
        if cost > diff:
            loc, cost = i, diff
    return loc, cost

def scan_mindiff(X, Sx, s0, s1):
    loc0, _ = find_mindiff(X, s0, s1)
    loc1, _ = find_mindiff(X, s1, s0)
    # print(s0.subs[loc0], s1.subs[loc1])
    if (loc0 == -1 or loc1 == -1
        or s0.subs[loc0, 1] < 2
        or s1.subs[loc1, 1] < 2):
        return np.inf
    tmp0 = Regime()
    tmp1 = Regime()
    st, ln = s0.subs[loc0]
    tmp0.add_segment(st, ln)
    st, ln = s1.subs[loc1]
    tmp1.add_segment(st, ln)
    _estimate_hmm_k(X, tmp0, MINK)
    _estimate_hmm_k(X, tmp1, MINK)
    costC = cut_point_search(X, Sx, tmp0, tmp1, False)
    del tmp0, tmp1
    return costC

def remove_noise_aux(X, Sx, s0, s1, per):
    if per == 0: return
    th = per * Sx.costT
    mprev = np.inf
    while mprev > s0.n_seg + s1.n_seg:
        mprev = s0.n_seg + s1.n_seg
        loc0, diff0 = find_mindiff(X, s0, s1)
        loc1, diff1 = find_mindiff(X, s1, s0)
        cost, idx = (diff0, 0) if diff0 < diff1 else (diff1, 1)
        if cost >= th:
            continue
        if idx == 0:
            st, dt = s0.del_segment(loc0)
            s1.add_segment(st, dt)
        else:
            st, dt = s1.del_segment(loc1)
            s0.add_segment(st, dt)

def remove_noise(X, Sx, s0, s1):
    if s0.n_seg <= 1 and s1.n_seg <= 1:
        return
    per = SEGMENT_R
    remove_noise_aux(X, Sx, s0, s1, per)
    costC = scan_mindiff(X, Sx, s0, s1)
    opt0 = Regime()
    opt1 = Regime()
    copy_segments(s0, opt0)
    copy_segments(s1, opt1)
    prev = np.inf
    while per <= SEGMENT_R * 10:
        if costC >= np.inf:
            break
        per *= 2
        remove_noise_aux(X, Sx, s0, s1, per)
        if s0.n_seg <= 1 or s1.n_seg <= 1:
            break
        costC = scan_mindiff(X, Sx, s0, s1)
        if prev > costC:
            copy_segments(s0, opt0)
            copy_segments(s1, opt1)
            prev = costC
        else:
            break
    copy_segments(opt0, s0)
    copy_segments(opt1, s1)
    # _estimate_hmm(X, s0)
    # _estimate_hmm(X, s1)
    del opt0, opt1

def copy_segments(s0, s1):  # from s0 to s1
    s1.subs = deepcopy(s0.subs)
    s1.n_seg = s0.n_seg
    s1.len = s0.len
    s1.costT = s0.costT
    s1.costC = s1.costC
    s1.delta = s0.delta

def select_largest(s):
    loc = np.argmax(s.subs[:, 1])
    st, dt = s.subs[loc]
    s.initialize()
    s.add_segment(st, dt)

def uniformset(X, Sx, n_samples, seedlen):
    u = Regime()
    w = int((Sx.len - seedlen) / n_samples)
    for i in range(Sx.n_seg):
        if u.n_seg >= n_samples:
            return u
        st, ln = Sx.subs[i]
        ed = st + ln
        for j in range(n_samples):
            nxt = st + j * w
            if nxt + seedlen > ed:
                st = ed - seedlen
                if st < 0: st = 0
                u.add_segment_ex(st, seedlen)
                break
            u.add_segment_ex(nxt, seedlen)
    return u

def fixed_sampling(X, Sx, seedlen):
    # print('nseg', Sx.n_seg)
    s0, s1 = Regime(), Regime()
    loc = 0 % Sx.n_seg
    r = Sx.subs[loc, 0]
    if Sx.n_seg == 1:
        dt = Sx.subs[0, 1]
        if dt < seedlen:
            s0.add_segment(r, dt)
            s1.add_segment(r, dt)
        else:
            s0.add_segment(r, dt)
            s1.add_segment(r, dt)
    s0.add_segment(r, seedlen)
    loc = 1 % Sx.n_seg
    r = Sx.subs[loc, 0] + int(Sx.subs[loc, 1] / 2)
    s1.add_segment(r, seedlen)
    return s0, s1

def uniform_sampling(X, Sx, length, n1, n2, u):
    s0, s1 = Regime(), Regime()
    i, j = int(n1 % u.n_seg), int(n2 % u.n_seg)
    # print(i, j)
    st0, st1 = u.subs[i, 0], u.subs[j, 0]
    if abs(st0 - st1) < length:
        return s0, s1
    s0.add_segment(st0, length)
    s1.add_segment(st1, length)
    return s0, s1

def _find_centroid_wrap(X, Sx, seedlen, idx0, idx1, u):
    s0, s1 = uniform_sampling(X, Sx, seedlen, idx0, idx1, u)
    if not s0.n_seg or not s1.n_seg:
        return np.inf, None, None
    subs0 = s0.subs[0]
    subs1 = s1.subs[0]
    _estimate_hmm_k(X, s0, MINK)
    _estimate_hmm_k(X, s1, MINK)
    cut_point_search(X, Sx, s0, s1, False)
    if not s0.n_seg or not s1.n_seg:
        return np.inf, None, None
    costT_s01 = s0.costT + s1.costT
    return costT_s01, subs0, subs1

def _find_centroid(X, Sx, n_samples, seedlen):
    u = uniformset(X, Sx, n_samples, seedlen)
    # print(u.subs[:u.n_seg], u.n_seg)

    if parallel is True:
        results = Parallel(n_jobs=4)(
            [delayed(_find_centroid_wrap)(X, Sx, seedlen, iter1, iter2, u)
            for iter1, iter2 in combinations(range(u.n_seg), 2)])
    else:
        results = []
        for iter1, iter2 in tqdm(combinations(range(u.n_seg), 2), desc='SearchCentroid'):
            results.append(_find_centroid_wrap(X, Sx, seedlen, iter1, iter2, u))

    # pp.pprint(results)
    if not results:
        print('fixed sampling')
        s0, s1 = fixed_sampling(X, Sx, seedlen)
        return s0, s1
    centroid = np.argmin([res[0] for res in results])
    # print(results[centroid])
    costMin, seg0, seg1 = results[centroid]
    if costMin == np.inf:
        print('!! --- centroid not found')
        # s0, s1 = fixed_sampling(X, Sx, seedlen)
        # print('fixed_sampling', s0.subs, s1.subs)
        return Regime(), Regime()
    s0, s1 = Regime(), Regime()
    s0.add_segment(seg0[0], seg0[1])
    s1.add_segment(seg1[0], seg1[1])
    # print(s0.n_seg, s1.n_seg)
    # time.sleep(3)
    return s0, s1


if __name__ == '__main__':

    X = np.loadtxt('./datasets/21_01.amc.4d')
    X = scale(X)
    ap = AutoPlait()

    start = time.time()
    ap.solver(X)
    elapsed_time = time.time() - start
    print(f'==> elapsed time:{elapsed_time} [sec]')

    ap.save()
