

import pymoo
import os

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.factory import get_decision_making
from pymoo.visualization.scatter import Scatter



import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.moead import MOEAD

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.factory import get_termination

from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population

class MyProblem(Problem):

    def __init__(self, distances_target, distances_dataset, classes, ratio):
        super().__init__(n_var=len(distances_target),
                         n_obj=4,
                         xl=np.zeros(len(distances_target)),
                         xu=np.ones(len(distances_target)))
        self.distances_target = distances_target
        self.distances_dataset = distances_dataset
        self.classes = np.array(classes)
        self.ratio = ratio

        positive_idexes = np.array([i for i, el in enumerate(self.classes) if el])
        negative_idexes = np.array([i for i, el in enumerate(self.classes) if not el])
        self.indexes = [negative_idexes,positive_idexes]

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = (x*self.distances_target).sum(axis=1)/self.classes.sum()
        f2 = 1-(x.dot(self.distances_dataset)*x).sum(axis=1)/self.classes.sum()**2
        #f3 = abs((x*self.classes).sum(axis=1) - self.ratio[1])
        #f4 = abs((x*(1-self.classes)).sum(axis=1) == self.ratio[0])
        g1 = (x*self.classes).sum(axis=1) == self.ratio[1]
        g2 = (x*(1-self.classes)).sum(axis=1) == self.ratio[0]


        out["F"] = np.column_stack([f1, f2])
        #out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

    """
    # --------------------------------------------------
    # Pareto-front - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_front(self, flatten=True, **kwargs):
        f1_a = np.linspace(0.1**2, 0.4**2, 100)
        f2_a = (np.sqrt(f1_a) - 1)**2

        f1_b = np.linspace(0.6**2, 0.9**2, 100)
        f2_b = (np.sqrt(f1_b) - 1)**2

        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
        return stack(a, b, flatten=flatten)

    # --------------------------------------------------
    # Pareto-set - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_set(self, flatten=True, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a,b, flatten=flatten)"""

from pymoo.algorithms.so_pattern_search import PatternSearch
def select_graphs(distances_target, distances_dataset, classes, ratio):

    problem = MyProblem(distances_target, distances_dataset, classes, ratio)

    X = np.random.random((300, problem.n_var))< (sum(ratio)/problem.n_var)
    pop = Population().new("X", X)
    Evaluator().eval(problem, pop)

    n_samples = 200


    algorithm = NSGA2(
        pop_size=n_samples,
        sampling=MySampling(),
        crossover=BinaryCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=True
    )



    termination = get_termination("n_gen", 250)



    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   verbose=True)
    viualize(problem, res)
    i= decision(res)
    return res.X[i]


from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            for cl in [0,1]:
                I = np.random.permutation(len(problem.indexes[cl]))[:problem.ratio[cl]]
                I = np.array([problem.indexes[cl][el] for el in I])
                X[k, I] = True
        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            for cl in [0, 1]:
                cl_els = np.logical_xor(not cl, problem.classes)
                both_are_true = np.logical_and(np.logical_and(p1, p2),cl_els)
                _X[0, k, both_are_true] = True
                n_remaining = problem.ratio[cl] - np.sum(both_are_true)
                I = np.where(np.logical_and(np.logical_xor(p1, p2), cl_els))[0]
                S = I[np.random.permutation(len(I))][:n_remaining]
                _X[0, k, S] = True
        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):

        for i in range(X.shape[0]):
            #swap two elements from each  class
            for cl in [0,1]:
                pos = [el for el in problem.indexes[cl] if X[i, el]]
                neg = [el for el in problem.indexes[cl] if not X[i, el]]

                X[i, np.random.choice(neg)] = True
                X[i, np.random.choice(pos)] = False
        return X

import matplotlib.pyplot as plt
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.performance_indicator.gd import GD

from pymoo.factory import get_performance_indicator

from pymoo.visualization.scatter import Scatter

def viualize(problem,res):
    # get the pareto-set and pareto-front for plotting
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # Design Space
    """plot = Scatter(title = "Design Space", axis_labels="x")
    plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.do()
    plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
    plot.apply(lambda ax: ax.set_ylim(-2, 2))
    plot.show()"""

    # Objective Space
    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.show()



    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]

    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]

    # create the performance indicator object with reference point (4,4)

    metric = Hypervolume(ref_point=np.array([1, 1]))
    # calculate for each generation the HV metric
    hv = [metric.calc(f) for f in obj_and_feasible_each_gen]
    # function evaluations at each snapshot
    n_evals = np.array([a.evaluator.n_eval for a in res.history])

    # visualze the convergence curve
    color = 'tab:blue'

    plt.xlabel("Function Evaluations")
    plt.ylabel('Hypervolume', color=color)
    plt.plot(n_evals, hv, '-o',color=color)
    plt.tick_params(axis='y', labelcolor=color)
    plt.title("Convergence")



    #plt.xlabel("Function Evaluations")
    #plt.ylabel("Hypervolume")


    plt.show()

from pymoo.factory import get_problem, get_visualization, get_decomposition
from pymoo.factory import get_decision_making, get_reference_directions
def decision(res, weights = np.array([1/140, -3])):
    pf = res.F

    dm = get_decision_making("pseudo-weights",weights)

    I = dm.do(pf)

    plot = Scatter(angle=(10, 140))
    plot.add(pf, alpha=0.2)
    plot.add(pf[I], color="red", s=100)
    plot.show()
    return I