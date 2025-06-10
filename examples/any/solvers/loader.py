import torch
import pickle

from ..solvers.solver import Solver

from ..solvers.burger import burger_pde, burger_dirichlet_generator, burger_colloc_generator
from ..solvers.schrodinger import schrodinger_pde, schrodinger_dirichlet_generator,schrodinger_periodic_generator, schrodinger_colloc_generator




def load_problem(equation: str):
    match equation:
        case "burger":
            picklefile = "examples/any/solvers/burger_solver.pkl"
            pde_func = burger_pde
            dirichlet_generator = burger_dirichlet_generator
            periodic_generator = lambda: (torch.zeros(2), torch.zeros(2))  # No periodic conditions for Burger's equation
            collocation_generator = burger_colloc_generator
            input_dim = 2  # (t, x)
            output_dim = 1

            n_dirichlet = 1
            n_periodic = 0
            n_colloc = 10

        case "schrodinger":
            picklefile = "examples/any/solvers/schrodinger_solver.pkl"
            pde_func = schrodinger_pde
            dirichlet_generator = schrodinger_dirichlet_generator
            periodic_generator = schrodinger_periodic_generator
            collocation_generator = schrodinger_colloc_generator

            input_dim = 2  # (t, x)
            output_dim = 2  # (u_real, u_imag)

            n_dirichlet = 1
            n_periodic = 1
            n_colloc = 10

        case _:
            raise ValueError(f"Unknown equation: {equation}")
        
    solver_pickle: Solver = pickle.load(open(picklefile, "rb"))

    return solver_pickle, pde_func, dirichlet_generator, periodic_generator, collocation_generator, input_dim, output_dim, n_dirichlet, n_periodic, n_colloc