from core.models.pinn import PINN
from core.datasets.pinn_dataset import PINNDataloader, dirichlet_type, periodic_type, colloc_type


from torch.optim import Optimizer
import torch

import time 


def to_batch(dirichlet: dirichlet_type, 
             periodic: periodic_type, 
             colloc: colloc_type, 
             
             device: torch.device):
    
    dirichlet = (dirichlet[0].to(device), dirichlet[1].to(device))
    periodic = (periodic[0].to(device), periodic[1].to(device))
    colloc = colloc.to(device)
    return dirichlet, periodic, colloc


def accuracy(model: PINN, test_loader: PINNDataloader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    total_MSE: float = 0.0
    n_total: int = 0

    with torch.no_grad():
        for _, (dirichlet, periodic, colloc) in enumerate(test_loader):
            dirichlet, periodic, colloc = to_batch(dirichlet, periodic, colloc, device)

            loss = model.loss(dirichlet, periodic, colloc)

            total_MSE += loss.item() * dirichlet[0].size(0)

            n_total += dirichlet[0].size(0)

    if n_total == 0:
        return 0.0
    
    return total_MSE / n_total




def train_one_epoch(model: PINN, 
                    train_loader: PINNDataloader, 
                    optimizer: Optimizer, 
                    device: torch.device):
    
    model.train()
    model.to(device)

    total_loss: float = 0.0
    total_dirichlet_loss: float = 0.0
    total_periodic_loss: float = 0.0
    total_pde_loss: float = 0.0

    n: int = 0
    n_dirichlet: int = 0
    n_periodic: int = 0
    n_colloc: int = 0

    for _, (dirichlet, periodic, colloc) in enumerate(train_loader):
        dirichlet, periodic, colloc = to_batch(dirichlet, periodic, colloc, device)

        optimizer.zero_grad()
        loss = model.loss(dirichlet, periodic, colloc)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * dirichlet[0].size(0)
        total_dirichlet_loss += model.dirichlet_loss_v.item() * dirichlet[0].size(0)
        total_periodic_loss += model.periodic_loss_v.item() * periodic[0].size(0)
        total_pde_loss += model.pde_loss_v.item() * colloc.size(0)
        n += dirichlet[0].size(0)
        n_dirichlet += dirichlet[0].size(0)
        n_periodic += periodic[0].size(0)
        n_colloc += colloc.size(0)
    
    
    avg_loss = total_loss / n if n > 0 else 0.0
    avg_dirichlet_loss = total_dirichlet_loss / n_dirichlet if n_dirichlet > 0 else 0.0
    avg_periodic_loss = total_periodic_loss / n_periodic if n_periodic > 0 else 0.0
    avg_pde_loss = total_pde_loss / n_colloc if n_colloc > 0 else 0.0
    return avg_loss, avg_dirichlet_loss, avg_periodic_loss, avg_pde_loss,


def train(model: PINN, 
          train_loader: PINNDataloader, 
          optimizer: Optimizer, 
          epochs: int,
          device: torch.device,
          test_loader: PINNDataloader | None = None, 
          verbose: bool = True):
    
    model.to(device)

    train_losses: list[float] = []
    train_dirichlet_losses: list[float] = []
    train_periodic_losses: list[float] = []
    train_pde_losses: list[float] = []

    test_losses: list[float] | None = [] if test_loader is not None else None

    times: list[float] = []

    test_loss = None

    for epoch in range(epochs):
        t0 = time.time()
        loss, dirichlet_loss, periodic_loss, pde_loss = train_one_epoch(model, train_loader, optimizer, device)
        times.append(time.time() - t0)

        train_losses.append(loss)
        train_dirichlet_losses.append(dirichlet_loss)
        train_periodic_losses.append(periodic_loss)
        train_pde_losses.append(pde_loss)

        if test_loader is not None:
            test_loss = accuracy(model, test_loader, device)
            test_losses.append(test_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Time: {times[-1]:.4f}s - "
                  f"Train Loss: {loss:.4f} - "
                  f"Dirichlet Loss: {dirichlet_loss:.4f} - "
                  f"Periodic Loss: {periodic_loss:.4f} - "
                  f"PDE Loss: {pde_loss:.4f} - ",
                  end="")
            if test_loss is not None:
                print(f"Test Loss: {test_loss:.4f} - ")


    return train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times