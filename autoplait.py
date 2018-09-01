import time
from copy import deepcopy
from itertools import combinations
from warnings import filterwarnings
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale
from sklearn.mixture import log_multivariate_normal_density
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed

cmap = cm.Set1
ZERO = 0 #1.e-10
INF = 1.e+10
MINK = 1
MAXK = 16
N_INFER_ITER_HMM = 1
INFER_ITER_MIN = 2
INFER_ITER_MAX = 10
SEGMENT_R = 1.e-2
REGIME_R = 1.e-2
# MAXBAUMN = 1000
FB = 4 * 8
LM = .1
NSAMPLE = 10
filterwarnings('ignore')

class AutoPlait():
    def __init__(self):
        self.costT = np.inf
        self.regimes = []

    def solver(self, X):
        self.X = X
        self.n, self.d = _shape(X)
        reg = Regime(_shape(X)[0])
        reg.add_segment(0, self.n)
        _estimate_hmm(X, reg)
        candidate = [reg]

        while True:
            self.costT = _mdl_total(self.regimes, candidate)
            reg = candidate.pop()
            # try to split regime: s0, s1
            reg0, reg1 = regime_split(X, reg)
            costT_s01 = reg0.costT + reg1.costT + REGIME_R * reg.costT
            print(f'\t-- try to split: {costT_s01:.6} vs {reg.costT:.6}')
            # print(s0.costT, s1.costT)
            if costT_s01 < reg.costT:
                candidate.append(reg0)
                candidate.append(reg1)
            else:
                self.regimes.append(reg)
            if not candidate: break

    def save(self):
        plt.subplot(211)
        plt.plot(self.X)
        plt.ylabel('Value')
        plt.subplot(212)
        for r in range(len(self.regimes)):
            for i in range(len(self.regimes[r].sub)):
                st, dt = self.regimes[r].sub[i]
                plt.plot([st, st + dt - 1], [r, r], color=cmap(r))
        plt.xlabel('Time')
        plt.ylabel('Regime ID')
        plt.tight_layout()
        plt.savefig('./result.png')
        plt.close()

def _mdl_total(stack0, stack1):
    r = len(stack0) + len(stack1)
    m = sum([regime.n_seg() for regime in stack0])
    m += sum([regime.n_seg() for regime in stack1])
    costT = MDLsegment(stack0) + MDLsegment(stack1)
    costT += log_s(r) + log_s(m) + m * np.log2(r) + FB * r ** 2
    print(f'[r, m, total_cost] = {r}, {m}, {costT:.6}')
    return costT

def regime_split(X, sx):
    opt0, opt1 = Regime(_shape(X)[0]), Regime(_shape(X)[0])
    n, d = _shape(X)
    seedlen = int(n * LM)
    s0, s1 = _find_centroid(X, sx, NSAMPLE, seedlen)
    if not s0.n_seg() or not s1.n_seg():
        return opt0, opt1
    for i in range(INFER_ITER_MAX):
        select_largest(s0)
        select_largest(s1)
        _estimate_hmm(X, s0)
        _estimate_hmm(X, s1)
        cut_point_search(X, sx, s0, s1)
        _compute_lh_mdl(X, s0)
        _compute_lh_mdl(X, s1)
        print(s0.sub)
        print(s1.sub)
        if not s0.n_seg() or not s1.n_seg(): break
        diff = (opt0.costT + opt1.costT) - (s0.costT + s1.costT)
        if diff > 0:
            copy_segments(s0, opt0)
            copy_segments(s1, opt1)
        elif i >= INFER_ITER_MIN: break
    copy_segments(opt0, s0)
    copy_segments(opt1, s1)
    del opt0, opt1
    if not s0.n_seg() or not s1.n_seg():
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
        Pv[v] += gaussian_pdfl(m0, v, X[t])
    Pj = np.log(d0) + np.log(m1.startprob_ + ZERO)
    for j in range(k1):
        Pj[j] += gaussian_pdfl(m1, j, X[t])

    for t in range(st + 1, st + dt):
        # Pu(t)
        maxj = np.argmax(Pj)
        for u in range(k0):
            maxPj = Pj[maxj] + np.log(d1) + np.log(m0.startprob_[u] + ZERO) + gaussian_pdfl(m0, u, X[t])
            val = Pv + np.log(1. - d0) + np.log(m0.transmat_[:, u] + ZERO)
            for v in range(k0):
                val[v] += gaussian_pdfl(m0, u, X[t])
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
            maxPv = Pv[maxv] + np.log(d0) + np.log(m1.startprob_[i] + ZERO) + gaussian_pdfl(m1, i, X[t])
            val = Pj + np.log(1. - d1) + np.log(m1.transmat_[:, i] + ZERO)
            for j in range(k1):
                val[j] += gaussian_pdfl(m1, i, X[t])
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
    return -llh / np.log(2.)  # data coding cost

def cut_point_search(X, sx, s0, s1, remove=True):
    s0.sub.clear()
    s1.sub.clear()
    lh = 0.
    for i in range(sx.n_seg()):
        lh += _search_aux(X, sx.sub[i][0], sx.sub[i][1], s0, s1)
    if remove: remove_noise(X, sx, s0, s1)
    return lh

def _mdl(regime):
    m = regime.n_seg()
    k = regime.model.n_components
    d = regime.model.n_features
    costT = costLen = 0.
    costC = regime.costC
    costM = costHMM(k, d)
    for i in range(m):
        costLen += np.log2(regime.sub[i][1])
    costLen += m * np.log2(k)
    return costC + costM + costLen

def _viterbi(X, hmm, delta):
    if not 0 <= delta <= 1:
        exit('not appropriate delta')
    llh = hmm.score(X) + np.log(delta) + np.log(1. - delta)
    return -llh / np.log(2.)  # data coding cost

def _compute_lh_mdl(X, regime):
    if regime.n_seg() == 0:
        regime.costT = regime.costC = np.inf
        return
    regime.costC = 0.
    for i in range(regime.n_seg()):
        st, dt = regime.sub[i]
        regime.costC += _viterbi(X[st:st+dt], regime.model, regime.delta)
    regime.costT = _mdl(regime)

def _shape(X):
    if np.ndim(X) > 1:
        n_sample, n_dim = X.shape
    else:
        n_sample, n_dim = len(X), 1
    return n_sample, n_dim

def _parse_input(X, regime):
    inputs = [X[st:st+dt] for st, dt in regime.sub]
    lengths = [dt for _, dt in regime.sub]
    return np.concatenate(inputs), lengths

def _estimate_hmm_k(X, regime, k=1):
    # lengths = [l for _, l in regime.sub]
    # idx = np.argsort(lengths)[::-1]  #
    # print('n_seg',regime.n_seg())
    # print(regime.sub)
    # if regime.n_seg() > MAXBAUMN:
    #     idx = idx[:MAXBAUMN]
    #     lengths = lengths[idx]
    # _X = [X[regime.sub[i][0]:regime.sub[i][0]+regime.sub[i][1]] for i in idx]
    # _X = np.concatenate(_X)
    # print('total length', np.sum(lengths))
    # print('# of samples', len(_X))
    _X, lengths = _parse_input(X, regime)
    # print(X.shape)
    # print(lengths)
    regime.model = GaussianHMM(n_components=k, covariance_type='diag',
                               n_iter=N_INFER_ITER_HMM)
    regime.model.fit(_X, lengths=lengths)
    regime.delta = regime.n_seg() / regime.len

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

class Regime():
    def __init__(self, end):
        self.sub = []
        self.len = 0
        self.costT = np.inf
        self.costC = np.inf
        self.model = None
        self.delta = 1.
        self.end = end
        # self.n_seg() = 0

    def n_seg(self):
        return len(self.sub)

    def add_segment(self, st, ln):
        if ln <= 0: return
        if st < 0: st = 0
        loc = 0
        for i in range(len(self.sub)):
            if self.sub[i][0] > st: break
            else: loc += 1
        self.sub.insert(loc, [st, ln])
        # remove overlap
        curr = INF
        while curr > len(self.sub):
            curr = len(self.sub)
            for i in range(curr - 1):
                st0, ln0 = self.sub[i]
                st1, ln1 = self.sub[i+1]
                ed0, ed1 = st0 + ln0, st1 + ln1
                ed = ed0 if ed0 > ed1 else ed1
                if ed0 + 1 >= st1:
                    self.sub.pop(i + 1)
                    self.sub[i][1] = ed - st0
                    break
        _, self.len = np.sum(np.array(self.sub), axis=0)
        self.delta = self.n_seg() / self.len

    def add_segment_ex(self, st, ln):
        self.len += ln
        self.sub.append((st, ln))

def log_s(x):
    return 2. * np.log2(x) + 1.

def costHMM(k, d):
    return FB * (k + k*k + 2*k*d) + 2. * np.log(k) / np.log(2.) + 1.

def MDLsegment(stack):
    return np.sum([regime.costT for regime in stack])

def gaussian_pdfl(hmm, state_id, x):
    mean = hmm.means_[state_id]
    covar = np.diag(hmm.covars_[state_id])
    n_dim = len(x)
    # x = np.expand_dims(x, 0)
    means = hmm.means_[0]
    covars = np.diag(hmm.covars_[0])
    return -.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars))
                  + np.sum((means ** 2) / covars)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1. / covars).T))

def find_mindiff(X, s0, s1):
    cost = np.inf
    loc = -1
    for i in range(len(s0.sub)):
        st, dt = s0.sub[i]
        costC0 = _viterbi(X[st:st+dt], s0.model, s0.delta)
        costC1 = _viterbi(X[st:st+dt], s1.model, s1.delta)
        diff = costC1 - costC0
        if cost > diff:
            loc, cost = i, diff
    return loc, cost

def scan_mindiff(X, Sx, s0, s1):
    loc0, _ = find_mindiff(X, s0, s1)
    loc1, _ = find_mindiff(X, s1, s0)
    if loc0 == -1 or loc1 == -1: return INF
    tmp0 = Regime(_shape(X)[0])
    tmp1 = Regime(_shape(X)[0])
    st, ln = s0.sub[loc0]
    tmp0.add_segment(st, ln)
    st, ln = s1.sub[loc1]
    tmp1.add_segment(st, ln)
    _estimate_hmm_k(X, tmp0, MINK)
    _estimate_hmm_k(X, tmp1, MINK)
    costC = cut_point_search(X, Sx, tmp0, tmp1, remove=False)
    del tmp0, tmp1
    return costC

def remove_noise_aux(X, Sx, s0, s1, per):
    if per == 0: return
    mprev = np.inf
    th = Sx.costT * per
    while mprev > len(s0.sub) + len(s1.sub):
        mprev = len(s0.sub) + len(s1.sub)
        loc0, diff0 = find_mindiff(X, s0, s1)
        loc1, diff1 = find_mindiff(X, s1, s0)
        cost, idx = (diff0, 0) if diff0 < diff1 else (diff1, 1)
        if cost >= th: continue
        if idx == 0:
            st, ln = s0.sub.pop(loc0)
            s1.add_segment(st, ln)
        else:
            st, ln = s1.sub.pop(loc1)
            s0.add_segment(st, ln)

def remove_noise(X, Sx, s0, s1):
    if s0.n_seg() <= 1 and s1.n_seg() <= 1: return
    per = SEGMENT_R
    remove_noise_aux(X, Sx, s0, s1, per)
    costC = scan_mindiff(X, Sx, s0, s1)
    opt0 = Regime(_shape(X)[0])
    opt1 = Regime(_shape(X)[0])
    copy_segments(s0, opt0)
    copy_segments(s1, opt1)
    prev = np.inf
    while per <= SEGMENT_R * 10:
        if costC >= INF: break
        per *= 2
        remove_noise_aux(X, Sx, s0, s1, per)
        if s0.n_seg() <= 1 or s1.n_seg() <= 1: break
        costC = scan_mindiff(X, Sx, s0, s1)
        if prev > costC:
            copy_segments(s0, opt0)
            copy_segments(s1, opt1)
        else: break
        prev = costC
    copy_segments(opt0, s0)
    copy_segments(opt1, s1)
    del opt0, opt1

def copy_segments(s0, s1):  # from s0 to s1
    s1 = deepcopy(s0)

def select_largest(s):
    loc = np.argmax(np.array(s.sub), axis=0)[1] 
    st, ln = s.sub[loc]
    s.sub.clear()
    s.add_segment(st, ln)

def uniformset(Sx, length, trial):
    u = Regime(_shape(X)[0])
    slide_w = int((Sx.len - length) / trial)
    for i in range(len(Sx.sub)):
        if len(u.sub) >= trial:
            return u
        st, ln = Sx.sub[i]
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

def fixed_sampling(X, Sx, length):
    # print('nseg', Sx.n_seg())
    s0, s1 = Regime(_shape(X)[0]), Regime(_shape(X)[0])
    loc = 0 % len(Sx.sub)
    r = Sx.sub[loc][0]
    if Sx.n_seg() == 1:
        _len = Sx.sub[0][1]
        if _len < length:
            s0.add_segment(r, _len)
            s1.add_segment(r, _len)
        else:
            s0.add_segment(r, _len)
            s1.add_segment(r, _len)
    s0.add_segment(r, length)
    loc = 1 % len(Sx.sub)
    r = Sx.sub[loc][0] + int(Sx.sub[loc][1] / 2)
    s1.add_segment(r, length)
    return s0, s1

def uniform_sampling(X, Sx, length, n1, n2, u):
    s0, s1 = Regime(_shape(X)[0]), Regime(_shape(X)[0])
    i, j = int(n1 % len(u.sub)), int(n2 % len(u.sub))
    st0, st1 = u.sub[i][0], u.sub[j][0]
    if abs(st0 - st1) < length:
        return s0, s1
    s0.add_segment(st0, length)
    s1.add_segment(st1, length)
    return s0, s1

def _find_centroid_wrap(X, Sx, seedlen, idx0, idx1, u):
    s0, s1 = uniform_sampling(X, Sx, seedlen, idx0, idx1, u)
    if not len(s0.sub) or not len(s1.sub):
        return np.inf, None, None
    sub0 = deepcopy(s0.sub[0])
    sub1 = deepcopy(s1.sub[0])
    _estimate_hmm_k(X, s0, MINK)
    _estimate_hmm_k(X, s1, MINK)
    cut_point_search(X, Sx, s0, s1, False)
    if not len(s0.sub) or not len(s1.sub):
        return np.inf, None, None
    _compute_lh_mdl(X, s0)
    _compute_lh_mdl(X, s1)
    costT_s01 = s0.costT + s1.costT
    return costT_s01, sub0, sub1

def _find_centroid(X, Sx, n_samples, seedlen):
    u = uniformset(Sx, seedlen, n_samples)
    # print('u', u.sub)
    n_sub = len(u.sub)
    result = Parallel(n_jobs=4)(
        [delayed(_find_centroid_wrap)(X, Sx, seedlen, iter1, iter2, u)
        for iter1, iter2 in combinations(range(n_sub), 2)])
    if not result:
        s0, s1 = fixed_sampling(X, Sx, seedlen)
        return s0, s1
    centroid = np.argmin([r[0] for r in result])
    costMin, seg0, seg1 = result[centroid]
    if costMin == np.inf:
        print('!! --- centroid not found')
        # s0, s1 = fixed_sampling(X, Sx, seedlen)
        # print('fixed_sampling', s0.sub, s1.sub)
        return Regime(0), Regime(0)
    s0, s1 = Regime(_shape(X)[0]), Regime(_shape(X)[0])
    s0.add_segment(seg0[0], seg0[1])
    s1.add_segment(seg1[0], seg1[1])
    return s0, s1


if __name__ == '__main__':

    X = np.loadtxt('./_dat/21_01.amc.4d')
    X = scale(X)
    ap = AutoPlait()
    start = time.time()
    ap.solver(X)
    elapsed_time = time.time() - start
    print(f'\t-- elapsed time:{elapsed_time:.2} [sec]')

    ap.save()