""" Python implementation of AutoPlait
"""

from copy import deepcopy
from itertools import combinations
import pickle
import numpy as np
from regime import Regime, concat_regimes, cp2seg
import utils


class AutoPlait():
    def __init__(self, n_sample=10, refinement=True,
                 float_cost=32, min_seglen=50, parallel=False,
                 seed=None, verbose=False):

        self.regimeset = []
        self.n_sample = n_sample
        self.split_rate = 0.1  # 0.1 => 10-fold
        self.float_cost = float_cost  # of bits
        self.min_seglen = min_seglen  #
        self.alpha_ap = 0.1  # coef. for AutoPlait
        self.alpha_rs = 0.01  # coef. for RegimeSplit

    def fit(self, data):
        """ Main loop
        """

        s0 = [0, data.shape[0]]  # whole input
        r0 = Regime(s0)
        r0 = r0.rfit(data)
        stack = [r0]  # to store candidate regimes

        while True:
            try:
                rx = stack.pop()
            except IndexError:
                # if the stack is empty,
                # pop() returns this error
                break  # convergence

            r1, r2 = self.rsplit(data, rx)
            print()
            print('{} VS {}'.format(
                r1.costT + r2.costT, rx.costT))

            # Convergence rule

            if r1.costT + r2.costT < rx.costT:
                print('SPLIT')
                stack.extend([r1, r2])
            else:
                print('NOT SPLIT')
                self.regimeset.append(rx)

    def rsplit(self, data, rx, n_iter_max=100):
        """ Inner loop: Regime Split

            data: array
                Original sequence
            rx: Regime object
                A candidate regime to split
        """
        print('RegimeSplit')
        r1, r2 = self.find_centroid(data, rx)
        # print('r1', r1.S)
        # print('r2', r2.S)
        cost = np.inf  # 

        for iteration in range(n_iter_max):
            # Cut point search
            S1, S2 = self.cps(data, rx, r1, r2)
            # print(S1)
            # print(S2)
            if len(S1) == 0 or len(S2) == 0:
                opt1 = Regime([0, 0])
                opt2 = Regime([0, 0])
                break
            # Parameter estimation
            r1 = Regime(S1).rfit(data)
            r2 = Regime(S2).rfit(data)

            cost12 = r1.costT + r2.costT
            print('Iter= {}\tMDL= {:.2f} (diff={:.2f})'.format(
                iteration + 1, cost12, cost - cost12))

            if cost12 < cost:
                opt1 = deepcopy(r1)
                opt2 = deepcopy(r2)
                cost = cost12
            else:
                break

        return opt1, opt2

    def cps(self, data, rx, r1, r2):
        """ Most inner loop: Cut Point Search
        """
        # print("CutPointSearch")
        # Concat two regimes
        model = concat_regimes(r1, r2)

        # Find the Viterbi path
        subs, lengths = rx.get_subsequence(data)
        states = model.predict(subs, lengths)

        states[states < r1.n_components] = 0
        states[states >= r1.n_components] = 1

        seg1, seg2 = cp2seg(rx.S, states)

        return seg1, seg2

    def find_centroid(self, data, rx):
        print('FindCentroid')
    
        # Choose the best pair that minimizes
        # total description cost for subsequences in rx

        seedlen = max(
            self.min_seglen,
            # int(self.split_rate * rx.S[:, 1].max())
            int(rx.S[:, 1].max()
                / int(np.log(rx.S[:, 1].max() + 1)))
        )
        # print('SEEDLEN=', seedlen)

        # S: list of subsequences
        S = utils.uniform_sampling(
            self.n_sample, seedlen, rx.S, topk=3)
        C = []  # List of model sets
        e = []  # List of total costs

        for s1, s2 in combinations(S, 2):
            r1 = Regime(s1).rfit(data)  # fit & compute MDL
            r2 = Regime(s2).rfit(data)  # fit & compute MDL
            C.append([r1, r2])
            e.append(r1.costT + r2.costT)

        r1, r2 = C[np.argmin(e)]  # best pair
        # print(r1.costT)
        # print(r2.costT)
        return r1, r2

    def compute_MDL(self, data):
        pass

    def save(self, fp, save_segment_only=False):
        if save_segment_only == True:
            for i, r in self.regimeset:
                np.savetxt(fp + f'segment.{i}.txt', r.S)
        else:
            with open(fp + 'result.pkl', 'wb') as f:
                pickle.dump(self, f)

    def load(self, fp):
        return self
