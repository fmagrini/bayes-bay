from parameters import Parameterization1D
from target import Target
from log_likelihood import LogLikelihood
from markov_chain import MarkovChain


param = Parameterization1D(5, (0,10), [[1.,10],[1.,10]])
targets = [
    Target("xx_wave", [1,2,3,4,5]), 
    Target("yy_wave", [3,4,5,6,7]), 
]

def fwd_dummy(proposed_model: dict):
    return proposed_model['voronoi_sites']

fwd_functions = [fwd_dummy, fwd_dummy]

mc = MarkovChain(param, targets, fwd_functions, 1)
mc.advance_chain(10)
    