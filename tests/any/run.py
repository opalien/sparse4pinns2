import torch
from torch import nn
import pickle
import os
import argparse
from examples.any.solvers.loader import load_problem
from examples.any.dataset import TrainAnyDataset, TestAnyDataset
from examples.any.model import AnyPINN 

from core.utils.train import train



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)






parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=2, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-e", "--epoch", type=int, default=10, help="Un nombre (défaut: 100)")

#args = parser.parse_args()

n = 16 #args.m_matrix**2
k = 5 # args.k_layers
epoch = 10000 #args.epoch
lr = 0.001
problem = "navier_stokes"#"schrodinger"#"burger" #args.problem


def visualize_burger(model: AnyPINN):
    from matplotlib import pyplot as plt
    model.eval()
    device = next(model.parameters()).device

    # on recharge les bornes et le solveur pour Burger
    solver, pde_func, bounds, input_dim, output_dim = load_problem(problem)
    (t_min, t_max), (x_min, x_max) = bounds[0], bounds[1]

    # grille d’évaluation
    Nt, Nx = 100, 200
    t = torch.linspace(t_min, t_max, Nt)
    x = torch.linspace(x_min, x_max, Nx)
    T, X = torch.meshgrid(t, x, indexing="ij")
    pts = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1).to(device)

    with torch.no_grad():
        u_pred = model(pts).cpu().numpy().reshape(Nt, Nx)
        u_exact = solver.func(pts).cpu().numpy().reshape(Nt, Nx)

    # affichage
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axs[0].pcolormesh(x.numpy(), t.numpy(), u_exact, shading="auto")
    fig.colorbar(im0, ax=axs[0])
    axs[0].set(title="Solution exacte", xlabel="x", ylabel="t")

    im1 = axs[1].pcolormesh(x.numpy(), t.numpy(), u_pred, shading="auto")
    fig.colorbar(im1, ax=axs[1])
    axs[1].set(title="Prédiction PINN", xlabel="x", ylabel="t")

    im2 = axs[2].pcolormesh(x.numpy(), t.numpy(), u_pred - u_exact,
                            shading="auto", cmap="seismic")
    fig.colorbar(im2, ax=axs[2])
    axs[2].set(title="Erreur", xlabel="x", ylabel="t")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    solver, pde_func, bounds, input_dim, output_dim = load_problem(problem)



    t_max = bounds[0][1]
    spatial_bounds = bounds[1:]

    train_dataset = TrainAnyDataset(
        solver.func,
        n_elements=1000,
        n_colloc=10000, 
        shape=spatial_bounds,
        t_max=t_max
    )

    test_dataset = TestAnyDataset(
        solver.func,
        n_elements=1000,
        shape=spatial_bounds,
        t_max=t_max
    )


    train_dataloader  = train_dataset.get_dataloader(100)
    test_dataloader = test_dataset.get_dataloader(100)
    
    layers = [
        nn.Linear(input_dim, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, output_dim),
    ]
    
    model = AnyPINN(layers, pde_func)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses_linear, _, _, test_losses_linear, times_linear = train(model, train_dataloader, optimizer, device, epoch, test_dataloader)
    print("model trained")

