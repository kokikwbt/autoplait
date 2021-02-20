import numpy as np
from hmmlearn import hmm


class Regime(hmm.GaussianHMM):
    def __init__(self, S, n_components=1):

        # Segments

        self.S = np.array(S)  # List of segments
        if self.S.ndim == 1:
            self.S = np.expand_dims(self.S, 0)
        self.total_length = 0
        self.costC = np.inf  # Coding cost
        self.costT = np.inf  # Total description cost

        # Parameters

        self.initialize(n_components)

    def initialize(self, n_components):

        try:
            del self.startprob_
            del self.transmat_
            del self.means_
            del self.covars_

        except:
            pass

        super().__init__(n_components=n_components,
                         covariance_type='diag',
                         min_covar=0.001,
                         algorithm='viterbi',
                         init_params='stmc')

        # self.delta = 1e-10  # zero
        self.delta = 1e-4
        # self.delta = 1 - 1e-10  # zero
        # self.delta = 1 - 1e-1

    def n_segment(self):
        return len(self.S)

    def add_segment(self, st, dt):
        """
            st: int
                Starting point of an inserted segment
            dt: int
                Length of an inserted segment
        """
        segment = [st, dt]  # not tuple
        self.S = np.vstack([self.S, segment])
        self.remove_overlap()

    def del_segment(self, loc):
        """
            loc: int
                Location (index) of a segment to be deleted
        """
        return

    def remove_overlap(self):
        """
        """
        # Sort by starting points
        self.S.sort(axis=0) 

        # Find overlapping segments

        # Marge overlapping segments

    def get_subsequence(self, data):
        """ Generate an input for hmm.GaussianHMM.fit
        """
        subs = np.vstack(
            [data[st:st+dt] for st, dt in self.S])
        # print(subs)
        lengths = self.S[:, 1]
        # print(lengths)

        return subs, lengths

    def rfit(self, data, mink=1, maxk=4, costF=32):

        subs, lengths = self.get_subsequence(data)
        cost = np.inf

        for k in range(mink, maxk + 1):

            # self = Regime(self.S, k)
            # self.initialize(k)
            # self.fit(subs, lengths)
            # self.compute_MDL(subs, lengths, costF)
            tmp = Regime(self.S, k)
            tmp.fit(subs, lengths)
            tmp.compute_MDL(subs, lengths, costF)

            if tmp.costT < cost:
                cost = tmp.costT
                # print('MDL=', cost)
            else:
                break

        optk = k - 1
        # Overhead
        # self.initialize(optk)
        # self.fit(subs, lengths)
        # self.compute_MDL(subs, lengths, costF)
        tmp = Regime(self.S, optk)
        tmp.fit(subs, lengths)
        tmp.compute_MDL(subs, lengths, costF)

        return tmp

    def compute_MDL(self, subs, lengths, costF):

        self.costC = self.compute_costC(subs, lengths)
        self.costM = self.compute_costM(costF)
        self.costT = self.costC + self.costM

        return self.costT

    def compute_costC(self, subs, lengths=None):
        return -1 * self.score(subs, lengths) / np.log(2)

    def compute_costM(self, costF):
        return (
            self.compute_costHMM(costF)
            + np.log(self.delta) + np.log(1 - self.delta)
        )

    def compute_costHMM(self, costF):

        k = self.n_components
        d = self.n_features

        return costF * (k + k ** 2 + 2 * k * d)


def concat_regimes(regime0, regime1):

    k0 = regime0.n_components
    k1 = regime1.n_components
    k = k0 + k1

    regime = hmm.GaussianHMM(n_components=k)

    regime.startprob_ = np.concatenate([
        regime0.startprob_, regime1.startprob_
    ])
    regime.startprob_ /= regime.startprob_.sum()

    regime.means_ = np.concatenate([
        regime0.means_, regime1.means_
    ])

    regime.covars_ = np.concatenate([
        [np.diag(cov) for cov in regime0.covars_],
        [np.diag(cov) for cov in regime1.covars_]
    ])

    # Define new transition matrix

    regime.transmat_ = np.zeros((k, k))

    # Inner switching

    regime.transmat_[:k0, :k0] = \
        regime0.transmat_ * (1 - regime0.delta)
    regime.transmat_[k0:, k0:] = \
        regime1.transmat_ * (1 - regime1.delta)

    # Outer switching
    if regime0.startprob_.ndim == 1:
        regime0.startprob_ = regime0.startprob_[:, np.newaxis]
    if regime1.startprob_.ndim == 1:
        regime1.startprob_ = regime1.startprob_[:, np.newaxis]

    regime.transmat_[k0:, :k0] = \
        regime0.delta * regime1.startprob_
    regime.transmat_[:k0, k0:] = \
        regime1.delta * regime0.startprob_

    regime.transmat_ = regime.transmat_
    regime.transmat_ /= regime.transmat_.sum(axis=0)
    regime.transmat_ = regime.transmat_.T

    return regime


def cp2seg(S, states):
    """
    """
    loc = 0
    seg1 = []
    seg2 = []

    for st, dt in S:
        _seg1, _seg2 = cp2seg_aux(
            st, dt, states[loc:loc + dt])

        # print()
        # print(st, st + dt)
        # print(_seg1)
        # print(_seg2)
        # print()
        seg1.append(_seg1)
        seg2.append(_seg2)
        loc += dt

    seg1 = np.vstack(seg1)
    seg2 = np.vstack(seg2)
    seg1 = seg1[~np.all(seg1 == 0, axis=1)]
    seg2 = seg2[~np.all(seg2 == 0, axis=1)]

    return seg1, seg2


def cp2seg_aux(st, dt, states):

    assign = np.where(np.diff(states == 0))[0]
    n_cp = len(assign)  # of cut points

    if n_cp == 0:
        seg1 = np.array([[st, st + dt]])
        seg2 = np.array([[0, 0]])

    elif n_cp == 1:
        seg1 = np.array([[st, st + assign[0]]])
        seg2 = np.array([[st + assign[0], st + dt]])

    elif n_cp == 2:
        seg1 = st + np.array([
            [0, assign[0]],
            [assign[-1], dt]
        ])
        seg2 = st + np.array([
            [assign[0], assign[1]]
        ])

    else:
        if n_cp % 2 == 0:
            # Regime 1
            seg1 = assign[1:-1].reshape((-1, 2))
            seg1 = np.vstack([[0, assign[0]], seg1])
            seg1 = np.vstack([seg1, [assign[-1], dt]])
            seg1 += st
            # Regime 2
            seg2 = seg1.flatten()
            seg2 = seg2.reshape((-1, 2))

        else:
            # Regime 1
            seg1 = assign[1:].reshape((-1, 2))
            seg1 = np.vstack([[0, assign[0]], seg1])
            seg1 += st
            # Regime 2
            seg2 = seg1.flatten()
            seg2 = seg2[1:-1].reshape((-1, 2))
            seg2 = np.vstack([seg2, [assign[-1], dt]])

        # Convert seg to a list of (st, dt)

    seg1[:, 1] -= seg1[:, 0]
    seg2[:, 1] -= seg2[:, 0]

    # print(seg1)
    # print(seg2) 

    return seg1, seg2
