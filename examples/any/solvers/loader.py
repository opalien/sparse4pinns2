import torch

import pickle
from ..solvers.burger import burger_pde
from ..solvers.schrodinger import schrodinger_pde

from ..solvers.solver import Solver

def load_problem(equation: str):
    match equation:
        case "burger":
            picklefile = "examples/any/solvers/burger_solver.pkl"
            pde_func = burger_pde
            bounds = [(0, 1), (-1, 1)]
            output_dim = 1
            input_dim = 2  # (t, x)

        case "schrodinger":
            picklefile = "examples/any/solvers/schrodinger_solver.pkl"
            pde_func = schrodinger_pde
            bounds = [(0, torch.pi/2.), (-5., 5.)]
            output_dim = 2
            input_dim = 2  # (t, x)

        case _:
            raise ValueError(f"Unknown equation: {equation}")
        

        
    
    solver_pickle: Solver = pickle.load(open(picklefile, "rb"))

    return solver_pickle, pde_func, bounds, input_dim, output_dim