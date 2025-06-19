import torch
from torch import nn

import os
import argparse
import random
import string
import json
import numpy as np


from core.utils.seed import set_seed


from examples.any.solvers.loader import load_problem
from examples.any.model import AnyPINN
from examples.any.dataset import AnyDataset

from core.utils.train import train, train_lbfgs
from core.utils.seed import set_seed
from core.utils.save import save_result, save_results

from core.layers.monarch import MonarchLinear



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=2, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-r", "--rep", type=int, default=10, help="Nombre de division (défaut: 5)")
parser.add_argument("-s", "--seed", type=int, default=0, help="Un nombre (défaut: 0)")
args = parser.parse_args()


set_seed(args.seed)
problem = args.problem




# p_dense = k_d*(m_d**2)**2
# p_monarch = 2*k_m*(m_m**3)

# let m_m = m_d = m
# p_monarch = p_dense
# => 2*k_m = k_d*m
# => k_m = k_d*m/2



k_d = args.k_layers
m = args.m_matrix
n = m**2
k_m_max = k_d * m // 2


K_m = np.linspace(k_d, k_m_max, args.rep, dtype=int).tolist()

lettres = string.ascii_letters
alea = f"{problem}_{args.seed}"
print(f"Séquence aléatoire générée: {alea}")
save_path = os.path.join("results", "lenght", f'results_{problem}_{alea}.json')


if __name__ == "__main__":
    f = 1000
    lr = 0.001
    adam_epochs = 15#1500
    lbfgs_epochs = 5#500



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
        n_dirichlet=f,
        n_periodic=0,
        n_colloc=0
    )

    train_dataloader  = train_dataset.get_dataloader(100)
    test_dataloader = test_dataset.get_dataloader(100)


    dense_layers = [
        nn.Linear(input_dim, n),
        *[nn.Linear(n, n) for _ in range(k_d)],
        nn.Linear(n, output_dim),
    ]

    dense_model = AnyPINN(
        layers=dense_layers,
        pde=pde_func,
    )


    optimizer = torch.optim.Adam(dense_model.parameters(), lr=lr)
    train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times = train(
        model=dense_model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        epochs=0,#adam_epochs,
        device=device,
        test_loader=test_dataloader,
        verbose=True
    )

    to_save = {
        "train_losses": train_losses,
        "train_dirichlet_losses": train_dirichlet_losses,
        "train_periodic_losses": train_periodic_losses,
        "train_pde_losses": train_pde_losses,
        "test_losses": test_losses,
        "times": times,
        "n": dense_model.layers[0].out_features,
        "k": len(dense_model.layers),
        "layers": str(dense_model.layers),
        "factor": "dense",
        "optimizer": "adam",
        "epoch": adam_epochs,
        "total_epoch": adam_epochs,
    }
    save_result(save_path, to_save)


#    optimizer = torch.optim.LBFGS(
#        dense_model.parameters(),
#        lr=1.0,
#        max_iter=10,
#        max_eval=20,
#        tolerance_grad=1e-7,
#        tolerance_change=1e-9,
#        history_size=150,
#        line_search_fn="strong_wolfe"
#    )
#    train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times = train_lbfgs(
#        model=dense_model,
#        train_loader=train_dataloader,
#        optimizer=optimizer,
#        device=device,
#        epochs=lbfgs_epochs,
#        test_loader=test_dataloader,
#        verbose=True
#    )
#
#    to_save = {
#        "train_losses": train_losses,
#        "train_dirichlet_losses": train_dirichlet_losses,
#        "train_periodic_losses": train_periodic_losses,
#        "train_pde_losses": train_pde_losses,
#        "test_losses": test_losses,
#        "times": times,
#        "n": dense_model.layers[0].out_features,
#        "k": len(dense_model.layers),
#        "layers": str(dense_model.layers),
#        "factor": "dense",
#        "optimizer": "lbfgs",
#        "epoch": lbfgs_epochs,
#        "total_epoch": adam_epochs+lbfgs_epochs,
#    }
#    save_result(save_path, to_save)





    for k_m in K_m:
        print(f"Training Monarch model with k_m = {k_m}")
        monarch_layers = [
            nn.Linear(input_dim, n),
            *[MonarchLinear(n, n) for _ in range(k_m)],
            nn.Linear(n, output_dim),
        ]

        monarch_model = AnyPINN(
            layers=monarch_layers,
            pde=pde_func,
        )

        optimizer = torch.optim.SGD(monarch_model.parameters())#torch.optim.Adam(monarch_model.parameters(), lr=lr)
        train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times = train(
            model=monarch_model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            epochs=1000, #adam_epochs,
            device=device,
            test_loader=test_dataloader,
            verbose=True
        )

        to_save = {
            "train_losses": train_losses,
            "train_dirichlet_losses": train_dirichlet_losses,
            "train_periodic_losses": train_periodic_losses,
            "train_pde_losses": train_pde_losses,
            "test_losses": test_losses,
            "times": times,
            "n": monarch_model.layers[0].out_features,
            "k": len(monarch_model.layers),
            "layers": str(monarch_model.layers),
            "factor": "monarch",
            "optimizer": "adam",
            "epoch": adam_epochs,
            "total_epoch": adam_epochs,
        }
        save_result(save_path, to_save)


#        optimizer = torch.optim.LBFGS(
#            monarch_model.parameters(),
#            lr=1.0,
#            max_iter=10,
#            max_eval=20,
#            tolerance_grad=1e-7,
#            tolerance_change=1e-9,
#            history_size=150,
#            line_search_fn="strong_wolfe"
#        )
#        train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times = train_lbfgs(
#            model=monarch_model,
#            train_loader=train_dataloader,
#            optimizer=optimizer,
#            device=device,
#            epochs=lbfgs_epochs,
#            test_loader=test_dataloader,
#            verbose=True
#        )
#
#        to_save = {
#            "train_losses": train_losses,
#            "train_dirichlet_losses": train_dirichlet_losses,
#            "train_periodic_losses": train_periodic_losses,
#            "train_pde_losses": train_pde_losses,
#            "test_losses": test_losses,
#            "times": times,
#            "n": monarch_model.layers[0].out_features,
#            "k": len(monarch_model.layers),
#            "layers": str(monarch_model.layers),
#            "factor": "monarch",
#            "optimizer": "lbfgs",
#            "epoch": lbfgs_epochs,
#            "total_epoch": adam_epochs+lbfgs_epochs,
#        }
#        save_result(save_path, to_save)
