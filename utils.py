import numpy as np


def uniform_sampling(n, w, S, topk=0):

    segments = np.sort(S, axis=1)[::-1]
    sample = []

    for st, dt in segments:
        for t in range(0, dt - w, w):
            sample.append([st + t, w])
            
            if len(sample) == n:
                break
        if len(sample) == n:
            break

    return np.array(sample)


def random_sampling(n, w, S, topk=0):
    return


def log_s(x):
    return 2 * np.log2(x) + 1
