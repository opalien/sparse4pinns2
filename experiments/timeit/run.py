import torch
from torch import nn

import os
import argparse
import random
import string
import json
import numpy as np

import timeit
import time
import sys

from matplotlib import pyplot as plt


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

lettres = string.ascii_letters
alea = ''.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}") 


save_path = os.path.join("results", "timeit", f'results_{alea}.json')


solver, pde_func, dirichlet_generator, periodic_generator, collocation_generator, test_generator, input_dim, output_dim, n_dirichlet, n_periodic, n_colloc = load_problem("burger")


number= 1000
K = [i for i in range(1, 60, 3)]
M = [i for i in range(2, 100, 3)]

for k in K:  
    t_dense = []
    t_monarch = []
    for m in M:
        print(f"Running for m={m}, k={k}")
        n = m**2
        dense_layers = [
            nn.Linear(2, n),
            *[MonarchLinear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]  

        monarch_layers = [
            nn.Linear(input_dim, n),
            *[MonarchLinear(n, n) for _ in range(k)],
            nn.Linear(n, output_dim),
        ]


        dense_model = AnyPINN(
            layers=dense_layers,
            pde=pde_func,
        ).to(device)

        monarch_model = torch.compile(AnyPINN(
            layers=monarch_layers,
            pde=pde_func,
        )).to(device)

        x = torch.randn(100, 2).to(device)


        def timeit_dense():
            dense_model(x)
        def timeit_monarch():
            monarch_model(x)

        
        t_d: float = timeit.timeit(timeit_dense, number=number, timer=time.process_time)
        t_m: float = timeit.timeit(timeit_monarch, number=number, timer=time.process_time)

        t_dense.append(t_d)
        t_monarch.append(t_m)

        print("t_dense", t_d)
        print("t_monarch", t_m)



    dict_to_save = {
        "m": list(range(2, 100)),
        "k": k,
        "t_dense": t_dense,
        "t_monarch": t_monarch
    }
    save_result(save_path, dict_to_save )


    plt.clf()
    plt.plot(M, t_dense, label="Dense")
    plt.plot(M, t_monarch, label="Monarch")
    #plt.plot(M, t_steam, label="STEAM")
    plt.xlabel("m (n = m*m)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"timeit_k={k}_alea={alea}.png")