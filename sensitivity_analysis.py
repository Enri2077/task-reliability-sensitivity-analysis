#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(linewidth=500, precision=4)
import sympy
from sympy import *
from sympy.matrices import Matrix as sympyMatrix
from sympy.matrices.dense import matrix2numpy
from apgl.graph import *

import sys
from collections import defaultdict


# Parameters and their vaules
parameters = dict()

p_N_S = Symbol('p_N_S')
parameters[p_N_S] = .9

p_GP_F2 = Symbol('p_GP_F2')
parameters[p_GP_F2] = .01

p_GP_F3 = Symbol('p_GP_F3')
parameters[p_GP_F3] = .05

p_GG_F2 = Symbol('p_GG_F2')
parameters[p_GG_F2] = .02

# Functionality outcome probability table: outcome_prob[F][O] = P[action has outcome O | action implements functionality F]
outcome_prob = defaultdict(dict)

outcome_prob['N']['S']   = p_N_S
outcome_prob['N']['F1']  = .9*(1 - p_N_S)
outcome_prob['N']['F2']  = .1*(1 - p_N_S)

outcome_prob['GT']['S']  = .6
outcome_prob['GT']['F1'] = .39
outcome_prob['GT']['F2'] = .01

outcome_prob['GP']['S']  = .9*(1 - p_GP_F2 - p_GP_F3)
outcome_prob['GP']['F1'] = .1*(1 - p_GP_F2 - p_GP_F3)
outcome_prob['GP']['F2'] = p_GP_F2
outcome_prob['GP']['F3'] = p_GP_F3

outcome_prob['GG']['S']  = .9*(1 - p_GG_F2)
outcome_prob['GG']['F1'] = .1*(1 - p_GG_F2)
outcome_prob['GG']['F2'] = p_GG_F2

# Build the Markov chain graph m and the parametric transition matrix P
N = 22
m = DenseGraph(N, undirected=False)
P = np.matrix(np.zeros((N, N)), dtype=sympy.symbol.Symbol)

# Nice function to create the graph node and set the element of the transition matrix at once
def edge(i, j, f=None, o=None):
    m[i, j] = 1
    if f is not None and o is not None:
        P[i, j] = outcome_prob[f][o]
    else:
        P[i, j] = 1

edge(0,  1)
edge(1,  2)
edge(2,  1,  'N', 'F1')
edge(2,  3,  'N', 'S')
edge(2,  15, 'N', 'F2')
edge(3,  4)
edge(4,  5)
edge(5,  4,  'GT', 'F1')
edge(5,  6,  'GT', 'S')
edge(5,  16, 'GT', 'F2')
edge(6,  7)
edge(7,  8)
edge(8,  9,  'N', 'F1')
edge(8,  10, 'N', 'S')
edge(8,  17, 'N', 'F2')
edge(9,  8)
edge(10, 11)
edge(11, 12)
edge(12, 11, 'GP', 'F1')
edge(12, 13, 'GP', 'S')
edge(12, 18, 'GP', 'F2')
edge(12, 19, 'GP', 'F3')
edge(13, 14)
edge(14, 14)
edge(19, 20)
edge(20, 13, 'GG', 'S')
edge(20, 19, 'GG', 'F1')
edge(20, 21, 'GG', 'F2')

x_i = 0  # task initial state
x_s = 14  # task success state

# Minimum distance from each node to the final state node
x_s_distance = m.floydWarshall()[:, x_s]

# List of state indices from which the state x_s is reachable. State x_s is excluded from this list. Cardinality r.
reaching_states = np.where(np.logical_and(x_s_distance != 0, x_s_distance < np.inf))[0]

# List of state indices from which the state x_s is unreachable.
non_reaching_states = np.where(x_s_distance == np.inf)[0]

# r×r sub-matrix of P, transition probability from each reaching state to each reaching state
P_1 = P[np.ix_(reaching_states, reaching_states)]

# r×1 sub-matrix of P, transition probability from each reaching state to state x_s
P_2 = P[np.ix_(reaching_states, [x_s])]

# Identity matrix
I = np.identity(len(reaching_states))

# Hitting probability from any reaching state to x_s, given by the system of linear equations h = h*P_1.T + P_2.T
h_reaching = P_2.T * matrix2numpy(sympyMatrix(I - P_1.T).inv())

h = np.zeros(len(P), dtype=sympy.symbol.Symbol)
h[reaching_states] = h_reaching
h[x_s] = 1.

# Probability of reaching x_s starting from x_i
h_i = h[x_i]

# Compute partial derivatites for each parameter
sensitivity = dict()
for p in parameters.keys():
    sensitivity[p] = diff(h_i, p).subs(parameters)
    print(str(p).ljust(10, ' '), '%0.5f' % float(sensitivity[p]))

if '-p' in sys.argv:
    from sympy.plotting import plot
    
    # Get parameter to be plotted from argument
    arg_index = sys.argv.index('-p')
    p1 = Symbol(sys.argv[arg_index + 1])
    not_p1 = list(set(parameters.keys()) - set([p1]))
    f = h_i
    
    # Substitute all parameters values except p1 in function h_i (renamed to f)
    for p in not_p1:
        f = f.subs(p, parameters[p])
        
    try:
        plot(f, xlim=(0, 1), ylim=(0, 1))
    except:
        print("Error while executing sympy.plotting.plot")

if '-s' in sys.argv:
    from sympy.plotting import plot3d_parametric_surface

    # Get the two parameters to be plotted from arguments
    arg_index = sys.argv.index('-s')
    p1, p2 = Symbol(sys.argv[arg_index + 1]), Symbol(sys.argv[arg_index + 2])
    not_p_1_p_2 = list(set(parameters.keys()) - set([p1, p2]))
    f = h_i
    
    # Substitute all parameters values except p1 and p2 in function h_i (renamed to f)
    for p in not_p_1_p_2:
        f = f.subs(p, parameters[p])

    try:
        plot3d_parametric_surface(p1, p2, f, (p1, 0, 1), (p2, 0, 1))
    except:
        print("Error while executing sympy.plotting.plot3d_parametric_surface")


if '-b' in sys.argv:
    # plot results
    import matplotlib.pyplot as plt

    objects = reaching_states
    probability = np.array(h_reaching)[0]

    y_pos = np.arange(len(objects))
    plt.bar(y_pos, probability, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Task Success Probability')
    plt.title('MC State')
    plt.ylim(0.9*np.min(probability), 1.)

    plt.show()
    
