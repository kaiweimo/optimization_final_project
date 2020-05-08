import numpy as np
import matplotlib.pyplot as plt
from dfdx import dfdx
from sympy import *
from sympy.parsing import sympy_parser as spp

def evolutionaty_strategy(m, obj, x_start, x_result, target_precision):
    xe = [[0.0, 0.0]]
    xe[0] = x_start
    iter_e = 0
    n_good_mutations = 0.0
    e_step = 2
    n = 10

    while true:
        for i in range(n):
            new_xe = np.random.normal(xe[-1], e_step, 2)
            iter_e += 1
            if obj.subs(m[0], new_xe[0]).subs(m[1], new_xe[1]) < obj.subs(m[0], xe[-1][0]).subs(m[1], xe[-1][1]):
                n_good_mutations += 1
                xe.append(new_xe)

        distance = np.linalg.norm(xe[-1] - x_result)
        if distance < target_precision:
            break  # stopping criterion
        if iter_e >= n:
            p_pos = n_good_mutations / iter_e
            n_good_mutations = 0.0
            if p_pos < 0.2:
                e_step *= 0.85
            else:
                e_step /= 0.85
            iter_e = 0
    xe = np.array(xe)
    print ("EVOLUTIONARY STRATEGY: distance:", distance)
    plt.plot(xe[:, 0], xe[:, 1], 'b-o')
