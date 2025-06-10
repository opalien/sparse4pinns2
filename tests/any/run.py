import torch
from torch import nn

import os
import argparse



from examples.any.solvers.loader import load_problem
from examples.any.model import AnyPINN
from examples.any.dataset import AnyDataset

from core.utils.train import train
from core.utils.seed import set_seed





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=2, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-e", "--epoch", type=int, default=10, help="Un nombre (défaut: 100)")
parser.add_argument("-s", "--seed", type=int, default=42, help="Un nombre (défaut: 42)")
#args = parser.parse_args()
set_seed(42) #args.seed
n = 16 #args.m_matrix**2
k = 5 # args.k_layers
epoch = 1000 #args.epoch
lr = 0.001
problem = "navier_stokes"#"schrodinger"#"burger" #args.problem




if __name__ == "__main__":
    f = 1000


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
        n_dirichlet=1000,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = train(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        epochs=epoch,
        device=device,
        test_loader=test_dataloader,
        verbose=True
    )
    


