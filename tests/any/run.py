import torch
from torch import nn
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np


import os
import argparse
from typing import Callable



from examples.any.solvers.loader import load_problem
from examples.any.model import AnyPINN
from examples.any.dataset import AnyDataset

from core.utils.train import train, train_lbfgs
from core.utils.seed import set_seed

from core.layers.monarch import MonarchLinear
from core.layers.steam import STEAMLinear



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
n = 25 #args.m_matrix**2
k = 5 # args.k_layers
epoch = 10 #args.epoch
lr = 0.001
problem = "schrodinger"#"navier_stokes"#"burger" #args.problem
alpha = 1.0
beta = 1.0#0.05





def plot_burger(func: Callable[[Tensor], Tensor]):
    import matplotlib.pyplot as plt
    import numpy as np
    t_lin = torch.linspace(0, 1, 256)
    x_lin = torch.linspace(-1, 1, 256)
    T, X = torch.meshgrid(t_lin, x_lin, indexing='ij')
    grid_points = torch.stack((T.flatten(), X.flatten()), dim=1).to(device)
    
    with torch.no_grad():
        U = func(grid_points).view(T.shape).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(U.T, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label="u(t, x)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()
    plt.show()


def plot_schrodinger(func: Callable[[Tensor], Tensor]):
    """
    Affiche le module de la solution de l'équation de Schrödinger.

    Args:
        func (Callable[[Tensor], Tensor]): Une fonction qui prend (t, x) et retourne (u_real, u_imag).
    """
    if isinstance(func, nn.Module):
        title_prefix = "Solution PINN"
        func.eval()
    else:
        title_prefix = "Solution exacte"

    t_lin = torch.linspace(0, torch.pi / 2, 256)
    x_lin = torch.linspace(-5, 5, 256)
    T, X = torch.meshgrid(t_lin, x_lin, indexing='ij')
    grid_points = torch.stack((T.flatten(), X.flatten()), dim=1).to(device)

    with torch.no_grad():
        output = func(grid_points)
        u_real = output[:, 0]
        u_imag = output[:, 1]
        h_mod = torch.sqrt(u_real**2 + u_imag**2).view(T.shape).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(h_mod.T, extent=[0, torch.pi / 2, -5, 5], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label="|h(t, x)|")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"{title_prefix} pour Schrödinger : Module |h|")
    plt.tight_layout()
    plt.show()


def plot_navier_stokes(func: Callable[[Tensor], Tensor], t_slice: float = 1.0):

    if isinstance(func, nn.Module):
        title_prefix = "Solution PINN"
        func.eval()
    else:
        title_prefix = "Solution exacte"

    x_lin = torch.linspace(0, 1, 128)
    y_lin = torch.linspace(0, 1, 128)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    
    t_grid = torch.full_like(X.flatten(), t_slice)
    grid_points = torch.stack((t_grid, X.flatten(), Y.flatten()), dim=1).to(device)

    with torch.no_grad():
        output = func(grid_points)
        psi = output[:, 0].view(X.shape).cpu().numpy()
        w = output[:, 1].view(X.shape).cpu().numpy()

    # Plot pour psi
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, psi.T, levels=50, cmap='viridis')
    plt.colorbar(label="$\psi(x, y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix} pour Navier-Stokes : $\psi$ à t={t_slice:.2f}")
    plt.axis('scaled')
    plt.show()

    # Plot pour w
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, w.T, levels=50, cmap='viridis')
    plt.colorbar(label="$w(x, y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix} pour Navier-Stokes : Vorticité $w$ à t={t_slice:.2f}")
    plt.axis('scaled')
    plt.show()




if __name__ == "__main__":
    f = 1000


    solver, pde_func, dirichlet_generator, periodic_generator, collocation_generator, test_generator, input_dim, output_dim, n_dirichlet, n_periodic, n_colloc = load_problem(problem)

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
        dirichlet_generator=test_generator,
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

#    layers = [
#        nn.Linear(input_dim, n),
#        *[MonarchLinear(n, n) for _ in range(k)],
#        nn.Linear(n, output_dim),
#    ]

    model = AnyPINN(
        layers=layers,
        pde=pde_func,
        alpha=alpha,
        beta=beta
    )



    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = train(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        epochs=10_000,#epoch,
        device=device,
        test_loader=test_dataloader,
        verbose=True
    )

    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, max_eval=20, tolerance_grad=1e-7, tolerance_change=1e-9)
    results_lbfgs = train_lbfgs(
        model=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=1000,
        test_loader=test_dataloader,
        verbose=True
    )
    


