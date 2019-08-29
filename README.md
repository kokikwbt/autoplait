# AutoPlait
[Automatic Mining of Co-evolving Time Sequences](http://www.cs.cmu.edu/~christos/PUBLICATIONS/14-sigmod-autoplait.pdf)

## Abstract
Given a large collection of co-evolving multiple time-series, which contains an unknown number of patterns of different durations, how can we efficiently and effectively find typical patterns and the points of variation? How can we statistically summarize all the sequences, and achieve a meaningful segmentation? In this paper we present AUTOPLAIT, a fully automatic mining algorithm for co-evolving time sequences. Our method has the following properties: (a) effectiveness: it operates on large collections of time-series, and finds similar segment groups that agree with human intuition; (b) scalability: it is linear with the in put size, and thus scales up very well; and (c) AUTOPLAIT is parameter-free, and requires no user intervention, no prior training, and no parameter tuning. Extensive experiments on 67GB of real datasets demonstrate that AUTOPLAIT does indeed detect meaningful patterns correctly, and it outperforms state-of-the-art competitors as regards accuracy and speed: AUTOPLAIT achieves near-perfect, over 95% precision and recall, and it is up to 472 times faster than its competitors.

## NOTE
- Now this code doesn't have smoothing algorithms.
