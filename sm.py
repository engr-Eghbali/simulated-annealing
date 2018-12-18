from __future__ import print_function, division  # Python 2 compatibility 





import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt 
import matplotlib as mpl
import math

from scipy import optimize       # to compare

import seaborn as sns
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

FIGSIZE = (19, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE



def annealing(random_start,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000,
              debug=True):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = [random_start(),random_start()]
    cost = cost_function(state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_state = random_neighbour(state, fraction)
        new_cost = cost_function(new_state)
        #print steps
        if debug: print("step: "+str(step),"\n maxspets: "+str(maxsteps),"\n temperature: "+str(T),"\n state: "+str(state),"\n cost: "+str(cost),"\n mewState: "+str(new_state),"\n newCost: "+str(new_cost)+"\n==============================\n")
        acceptance_probability(cost, new_cost, T)
        if acceptance_probability(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
    return state, cost_function(state), states, costs




interval = (-10, 10)

def f(x):
    """ Function to minimize."""
    X=x[0]
    Y=x[1]
    tmp=-0.0001*(np.absolute(np.sin(X)*np.sin(Y)*np.exp(np.absolute(100-math.sqrt(X**2+Y**2))+1)))**0.1
    return tmp

def clip(x):
    """ Force x to be in the interval."""
    a, b = interval
    return [max(min(x[0], b), a),max(min(x[1], b), a)]


def random_start():
    """ Random point in the interval."""
    a, b = interval
    return a + (b - a) * rn.random_sample()



def cost_function(x):
    """ Cost of x = f(x)."""
    return f(x)


def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    delta = (-amplitude/2.) + amplitude * rn.random_sample()
    tmp=[x[0]+delta,x[1]+delta]
    return clip(tmp)





def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        print(" \n   - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        print(" \n   - Acceptance probabilty = {:.3g}...".format(p))
        return p




def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))



annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=30, debug=True);




state, c, states, costs = annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=1000, debug=False)

state
c




def see_annealing(states, costs):
    plt.figure()
    plt.suptitle("Evolution of states and costs of the simulated annealing")
    plt.subplot(121)
    plt.plot(states, 'r')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'b')
    plt.title("Costs")
    plt.show()





see_annealing(states, costs)




def visualize_annealing(cost_function):
    state, c, states, costs = annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=1000, debug=False)
    see_annealing(states, costs)
    return state, c
