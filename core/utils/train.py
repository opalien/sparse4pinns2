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





def train_one_epoch_lbfgs(model: PINN, 
                          train_loader: PINNDataloader, 
                          optimizer: Optimizer, 
                          device: torch.device):
    
    model.train()
    model.to(device)

    # Dictionnaire pour stocker les métriques de la dernière évaluation de la closure.
    # Nécessaire car optimizer.step() peut appeler la closure plusieurs fois.
    last_closure_metrics = {
        'total_loss': 0.0,
        'dirichlet_loss': 0.0,
        'periodic_loss': 0.0,
        'pde_loss': 0.0,
        'n_dirichlet': 0,
        'n_periodic': 0,
        'n_colloc': 0
    }

    def closure():
        optimizer.zero_grad()
        
        # Pour une étape L-BFGS, nous accumulons la perte sur tous les batchs du loader.
        # Cela garantit que la fonction de perte est déterministe, une exigence pour L-BFGS.
        total_loss_for_step = torch.tensor(0.0, device=device)
        
        # Réinitialisation des accumulateurs pour le logging
        current_dirichlet_loss = 0.0
        current_periodic_loss = 0.0
        current_pde_loss = 0.0
        current_n_dirichlet = 0
        current_n_periodic = 0
        current_n_colloc = 0

        for _, (dirichlet, periodic, colloc) in enumerate(train_loader):
            dirichlet, periodic, colloc = to_batch(dirichlet, periodic, colloc, device)
            
            loss = model.loss(dirichlet, periodic, colloc)

            # Vérification de stabilité : arrête l'entraînement si la perte devient invalide.
            if torch.isnan(loss) or torch.isinf(loss):
                 raise ValueError(f"LBFGS: Perte invalide ({loss.item()}) détectée. L'entraînement ne peut pas continuer.")

            total_loss_for_step = total_loss_for_step + loss
            
            # Pour le logging, on suit la somme des erreurs au carré (non moyennée)
            if model.dirichlet_loss_v is not None and dirichlet[0].size(0) > 0:
                current_dirichlet_loss += model.dirichlet_loss_v.item() * dirichlet[0].size(0)
                current_n_dirichlet += dirichlet[0].size(0)
            if model.periodic_loss_v is not None and periodic[0].size(0) > 0:
                current_periodic_loss += model.periodic_loss_v.item() * periodic[0].size(0)
                current_n_periodic += periodic[0].size(0)
            if model.pde_loss_v is not None and colloc.size(0) > 0:
                current_pde_loss += model.pde_loss_v.item() * colloc.size(0)
                current_n_colloc += colloc.size(0)
        
        if total_loss_for_step.requires_grad:
            total_loss_for_step.backward()
        
        # On met à jour les métriques globales avec les valeurs de cette exécution de la closure
        nonlocal last_closure_metrics
        last_closure_metrics['total_loss'] = total_loss_for_step.item()
        last_closure_metrics['dirichlet_loss'] = current_dirichlet_loss
        last_closure_metrics['periodic_loss'] = current_periodic_loss
        last_closure_metrics['pde_loss'] = current_pde_loss
        last_closure_metrics['n_dirichlet'] = current_n_dirichlet
        last_closure_metrics['n_periodic'] = current_n_periodic
        last_closure_metrics['n_colloc'] = current_n_colloc
        
        return total_loss_for_step

    optimizer.step(closure)
    
    n_dirichlet = last_closure_metrics['n_dirichlet']
    n_periodic = last_closure_metrics['n_periodic']
    n_colloc = last_closure_metrics['n_colloc']
    
    # La perte totale de la closure est une somme de MSE par batch. On la moyenne par le nombre de batchs.
    avg_loss = last_closure_metrics['total_loss'] / len(train_loader) if len(train_loader) > 0 else 0.0
    # Les pertes individuelles sont des sommes pondérées, on les divise par leurs comptes respectifs pour obtenir la MSE.
    avg_dirichlet_loss = last_closure_metrics['dirichlet_loss'] / n_dirichlet if n_dirichlet > 0 else 0.0
    avg_periodic_loss = last_closure_metrics['periodic_loss'] / n_periodic if n_periodic > 0 else 0.0
    avg_pde_loss = last_closure_metrics['pde_loss'] / n_colloc if n_colloc > 0 else 0.0
    
    return avg_loss, avg_dirichlet_loss, avg_periodic_loss, avg_pde_loss


def train_lbfgs(model: PINN, 
                train_loader: PINNDataloader, 
                optimizer: Optimizer, 
                epochs: int,
                device: torch.device,
                test_loader: PINNDataloader | None = None, 
                verbose: bool = True):
    
    if not isinstance(optimizer, torch.optim.LBFGS):
        raise TypeError("L'optimiseur pour train_lbfgs doit être une instance de torch.optim.LBFGS.")

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
        
        try:
            loss, dirichlet_loss, periodic_loss, pde_loss = train_one_epoch_lbfgs(model, train_loader, optimizer, device)
            times.append(time.time() - t0)

            train_losses.append(loss)
            train_dirichlet_losses.append(dirichlet_loss)
            train_periodic_losses.append(periodic_loss)
            train_pde_losses.append(pde_loss)

            if test_loader is not None:
                test_loss = accuracy(model, test_loader, device)
                if test_losses is not None:
                    test_losses.append(test_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Time: {times[-1]:.4f}s - "
                      f"Train Loss: {loss:.4e} - "
                      f"Dirichlet Loss: {dirichlet_loss:.4e} - "
                      f"Periodic Loss: {periodic_loss:.4e} - "
                      f"PDE Loss: {pde_loss:.4e} - ",
                      end="")
                if test_loss is not None:
                    print(f"Test Loss: {test_loss:.4e}")
                else:
                    print()
        
        except ValueError as e:
            print(f"\nL'époque {epoch + 1}/{epochs} a échoué avec l'erreur : {e}")
            # Ajoute NaN pour indiquer l'échec et arrête la boucle
            times.append(time.time() - t0)
            train_losses.append(float('nan'))
            train_dirichlet_losses.append(float('nan'))
            train_periodic_losses.append(float('nan'))
            train_pde_losses.append(float('nan'))
            if test_loader is not None and test_losses is not None:
                test_losses.append(float('nan'))
            break

    return train_losses, train_dirichlet_losses, train_periodic_losses, train_pde_losses, test_losses, times