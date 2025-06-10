import torch
from torch import nn

import os
import argparse
import random
import string
import json


from core.utils.seed import set_seed


from examples.any.solvers.loader import load_problem
from examples.any.model import AnyPINN
from examples.any.dataset import AnyDataset

from experiments.tree.execution_tree import ExecutionTree





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=2, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-l", "--language", type=str, default="", help="path to the language o the learning")
parser.add_argument("-s", "--seed", type=int, default=42, help="Un nombre (défaut: 42)")
parser.add_argument("-f", "--factor", type=int, default=1000, help="Un nombre (défaut: 1000)")
args = parser.parse_args()

set_seed(args.seed)

n = args.m_matrix**2
k = args.k_layers
lr = 0.001
problem = args.problem
f = args.factor

list_language: list[list[str]] = json.load(open(args.language, "r"))["bests"] if args.language else []
language = lambda x: x in [element[:len(x)] for element in list_language] if list_language else lambda x: True


lettres = string.ascii_letters
alea = str(args.seed) #'monoid_'.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}")
save_path = os.path.join("results", "tree", f'results_{problem}_{alea}.json')


if __name__ == "__main__":
    solver, pde_func, dirichlet_generator, periodic_generator, collocation_generator, input_dim, output_dim, n_dirichlet, n_periodic, n_colloc = load_problem(problem)

    n_dirichlet, n_periodic, n_colloc = n_dirichlet * f, n_periodic * f, n_colloc * f

    train_dataset = AnyDataset(
        dirichlet_generator=dirichlet_generator,
        periodic_generator=periodic_generator,
        colloc_generator=collocation_generator,
        n_dirichlet=n_dirichlet,
        n_periodic=n_periodic,
        n_colloc=n_colloc
    )

    test_dataset = AnyDataset(
        dirichlet_generator=dirichlet_generator,
        periodic_generator=periodic_generator,
        colloc_generator=collocation_generator,
        n_dirichlet=f,
        n_periodic=0,
        n_colloc=0 
    )

    train_dataloader  = train_dataset.get_dataloader(100)
    test_dataloader = test_dataset.get_dataloader(100)   

    layers = [
        nn.Linear(input_dim, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, output_dim),
    ]

    model = AnyPINN(
        layers=layers,
        pde=pde_func,
    )

    tree = ExecutionTree(
        pinn=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        work_dir=os.path.join("results", "tree"),
        steps=[i*500 for i in range(4+1)],
        alea=alea,
        language=language
    )

    tree.run()