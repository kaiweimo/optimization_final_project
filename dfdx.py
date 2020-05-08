def dfdx(x, g, m):
    return [float(g[i].subs(m[0], x[0]).subs(m[1], x[1])) for i in range(len(g))]