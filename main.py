import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.parsing import sympy_parser as spp
from gradient_descent import *
from newton_method import *
from evolutionaty_strategy import *

plot_from, plot_to, plot_step = -7.0, 7.0, 0.1
target_precision = 0.3    # distance between real results and computed results
m = Matrix(symbols('x1 x2'))

# draw pictures
start = [-4.0, -5.0]
obj = spp.parse_expr('(1 - x1)**2 + 10 * (x2 - x1**2)**2')
result = np.array([1, 1])
i1 = np.arange(plot_from, plot_to, plot_step)
i2 = np.arange(plot_from, plot_to, plot_step)
x1_mesh, x2_mesh = np.meshgrid(i1, i2)
f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
f_mesh = eval(f_str)
plt.figure()
plt.imshow(f_mesh, cmap='Paired', origin='lower', extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
plt.colorbar()
plt.title('f(x) = ' + str(obj))
plt.xlabel('x1')
plt.ylabel('x2')
newton_method(m, obj, result, start)
gradient_descent(m, obj, start, result, alpha=0.001)
evolutionaty_strategy(m, obj, start, result, target_precision)
plt.show()

#another picture
start = [-4.0, 6.0]
obj = spp.parse_expr('x1**2 - 2 * x1 * x2 + 4 * x2**2')
result = np.array([0, 0])
i1 = np.arange(plot_from, plot_to, plot_step)
i2 = np.arange(plot_from, plot_to, plot_step)
x1_mesh, x2_mesh = np.meshgrid(i1, i2)
f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
f_mesh = eval(f_str)
plt.figure()
plt.imshow(f_mesh, cmap='Paired', origin='lower', extent=[plot_from - 3, plot_to + 3, plot_from - 3, plot_to + 3])
plt.colorbar()
plt.title('f(x) = ' + str(obj))
plt.xlabel('x1')
plt.ylabel('x2')
newton_method(m, obj, result, start)
gradient_descent(m, obj, start, result, alpha=0.05)
evolutionaty_strategy(m, obj, start, result, target_precision)
plt.show()

