#!/usr/bin/env python2.6

"""Softmax."""

import numpy as np
import matplotlib.pyplot as plt

scores1 = [30.0, 10.0, 2.0]
scores2 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x)
    return np.divide(exp_x, np.sum(exp_x, axis=0))


print(softmax(scores1))
print(softmax(scores2))

# Plot softmax curves

x = np.arange(-2.0, 6.0, 1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
