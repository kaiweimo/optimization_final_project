import numpy as np
import matplotlib.pyplot as plt
from dfdx import dfdx
from sympy import *
from sympy.parsing import sympy_parser as spp

def gradient_descent(m, obj, x_start, x_result, alpha=0.002):
    g = [diff(obj, i) for i in m]
    xs = [[0.0, 0.0]]
    xs[0] = x_start

    iter_s = 0
    for iter_s in range(100):
        gs = dfdx(xs[iter_s], g, m)
        xs.append(xs[iter_s] - np.dot(alpha, gs))
        # print xs[-1]
        iter_s += 1
    print ("GRADIENT DESCENT: result distance:", np.linalg.norm(xs[-1] - x_result))
    xs = np.array(xs)
    plt.plot(xs[:, 0], xs[:, 1], 'g-o')