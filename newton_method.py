import numpy as np
import matplotlib.pyplot as plt
from dfdx import dfdx
from sympy import *
from sympy.parsing import sympy_parser as spp

def newton_method(m, obj, x_result, x_start):
    g = [diff(obj, i) for i in m]
    H = Matrix([[diff(g[j], m[i]) for i in range(len(m))] for j in range(len(g))])  # Hessian matrix
    H_inv = H.inv()
    tmp = np.array(x_result)
    xn = [[0, 0]]
    xn[0] = x_start
    tmp_target = 0.0

    iter_n = 0
    for iter_n in range(100):
        gn = Matrix(dfdx(xn[iter_n], g, m))
        delta_xn = -H_inv * gn
        delta_xn = delta_xn.subs(m[0], xn[iter_n][0]).subs(m[1], xn[iter_n][1])
        xn.append(Matrix(xn[iter_n]) + delta_xn)
        tmp = tmp.reshape((2,1))
        iter_n += 1
    print ("NEWTON METHOD: result distance:", np.linalg.norm(np.array(xn[-1], dtype=np.float64) - tmp))
    xn = np.array(xn)
    plt.plot(xn[:, 0], xn[:, 1], 'k-o')

