from core.models.pinn import PINN
from core.datasets.pinn_dataset import PINNDataloader


from torch.optim import Optimizer
from torch.nn import Module
import torch
from torch import Tensor
from torch.optim.lbfgs import LBFGS

import time 


def accuracy(model: Module, test_loader: PINNDataloader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    total_MSE: float = 0.0
    num_batches: int = 0
    with torch.no_grad():
        for i, (a, u, idx) in enumerate(test_loader): # type: ignore
            
            a: Tensor = a.float().to(device)
            u: Tensor = u.float().to(device)
    

            model(a)
            MSE = model.loss(u)

            total_MSE += MSE.item()
            num_batches += 1
    
    return total_MSE / num_batches if num_batches > 0 else 0.0


def train_one_epoch_lbfgs(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device):
    model.train()
    model.to(device)

    _pde_loss_sum_last_closure = 0.0
    _data_loss_sum_last_closure = 0.0
    _batches_in_last_closure = 0

    def closure():
        nonlocal _pde_loss_sum_last_closure, _data_loss_sum_last_closure, _batches_in_last_closure
        optimizer.zero_grad()
        
        total_loss_for_closure: Tensor | None = None
        
        current_closure_pde_loss_sum = 0.0
        current_closure_data_loss_sum = 0.0
        current_closure_num_batches = 0

        for i, batch_data_tuple in enumerate(train_loader): 
            a, u, idx_val = batch_data_tuple
            a = a.float().to(device)
            u = u.float().to(device)
            
            idx_for_loss: int | None = None
            if isinstance(idx_val, torch.Tensor):
                idx_for_loss = int(idx_val.item()) if idx_val.numel() == 1 else None
            elif isinstance(idx_val, int):
                idx_for_loss = idx_val

            model(a) # Forward pass
            loss = model.loss(u, idx_for_loss) 
            pde_loss = model.get_pde_loss()    
            data_loss = model.get_data_loss()  

            if torch.isnan(loss) or torch.isinf(loss):
                 raise ValueError(f"LBFGS: Invalid loss detected in closure: {float(loss.item())}.")
            
            if total_loss_for_closure is None:
                total_loss_for_closure = loss
            else:
                total_loss_for_closure = total_loss_for_closure + loss

            current_closure_pde_loss_sum += pde_loss.item()
            current_closure_data_loss_sum += data_loss.item()
            current_closure_num_batches += 1
        
        if current_closure_num_batches == 0 or total_loss_for_closure is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if total_loss_for_closure.requires_grad:
            total_loss_for_closure.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        _pde_loss_sum_last_closure = current_closure_pde_loss_sum
        _data_loss_sum_last_closure = current_closure_data_loss_sum
        _batches_in_last_closure = current_closure_num_batches
        
        return total_loss_for_closure

    final_total_loss_tensor: Tensor = optimizer.step(closure) 
    
    if _batches_in_last_closure > 0:
        avg_total_loss = float(final_total_loss_tensor.item()) / _batches_in_last_closure # CORRECTED
        avg_pde_loss = _pde_loss_sum_last_closure / _batches_in_last_closure
        avg_data_loss = _data_loss_sum_last_closure / _batches_in_last_closure
    else:
        avg_total_loss = 0.0
        avg_pde_loss = 0.0
        avg_data_loss = 0.0
    
    return avg_data_loss + model.lmda * avg_pde_loss, avg_pde_loss, avg_data_loss


def train_lbfgs(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device, epochs: int, test_loader: PINNDataloader | None = None, verbose: bool = True):
    if not isinstance(optimizer, LBFGS):
        raise TypeError("Optimizer for train_lbfgs must be an instance of torch.optim.LBFGS.")

    model.to(device)

    history_train_loss: list[float] = []
    history_pde_loss: list[float] = []
    history_data_loss: list[float] = []
    # history_reg_loss: list[float] = [] # If you add regularization reporting
    history_test_loss: list[float] | None = [] if test_loader is not None else None
    epoch_times: list[float] = []

    for epoch in range(epochs):
        t_start = time.time()
        
        # Call train_one_epoch_lbfgs for LBFGS optimizer
        # train_one_epoch_lbfgs returns: avg_total_loss_epoch, avg_pde_loss_epoch, avg_data_loss_epoch
        avg_total_loss, avg_pde_loss, avg_data_loss = train_one_epoch_lbfgs(model, train_loader, optimizer, device) # Potentially add avg_reg_loss
        
        t_end = time.time()
        epoch_times.append(t_end - t_start)

        history_train_loss.append(avg_total_loss)
        history_pde_loss.append(avg_pde_loss)
        history_data_loss.append(avg_data_loss)
        # history_reg_loss.append(avg_reg_loss) # If you add it

        log_message = f"Epoch {epoch+1}/{epochs} | Time: {epoch_times[-1]:.2f}s"
        log_message += f" | LBFGS Train Loss: {avg_total_loss:.6e}" # Clarified it's LBFGS
        log_message += f" | PDE Loss: {avg_pde_loss:.6e}"
        log_message += f" | Data Loss: {avg_data_loss:.6e}"
        # log_message += f" | Reg Loss: {avg_reg_loss:.2e}" # If you add it

        if history_test_loss is not None and test_loader is not None:
            current_test_loss = accuracy(model, test_loader, device)
            history_test_loss.append(current_test_loss)
            log_message += f" | Test Loss: {current_test_loss:.6e}"

        if verbose:
            print(log_message)
            
    return history_train_loss, history_pde_loss, history_data_loss, history_test_loss, epoch_times # Potentially history_reg_loss


def train_one_epoch(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device):
    model.train()
    model.to(device)

    total_loss: float = 0.0
    total_pde_loss: float = 0.0
    total_data_loss: float = 0.0

    num_batches: int = 0
    for i, (a, u, idx) in enumerate(train_loader): # type: ignore
        a: Tensor = a.float().to(device)
        u: Tensor = u.float().to(device)
        optimizer.zero_grad()

        model(a)

        loss = model.loss(u, idx)
        pde_loss = model.get_pde_loss()
        data_loss = model.get_data_loss()

        
        if torch.isnan(loss) or torch.isinf(loss):
            if torch.isnan(pde_loss) or torch.isinf(pde_loss):
                raise ValueError(f"Warning: Invalid PDE loss detected: {pde_loss.item()}. Skipping batch.")
            if torch.isnan(data_loss) or torch.isinf(data_loss):
                raise ValueError(f"Warning: Invalid data loss detected: {data_loss.item()}. Skipping batch.")
            raise ValueError(f"Warning: Invalid loss detected: {loss.item()}. Skipping batch.")
            
            
        
        loss.backward()
        
        
        optimizer.step()

        total_loss += loss.item()
        total_pde_loss += pde_loss.item()
        total_data_loss += data_loss.item()
        num_batches += 1


    return (total_loss / num_batches, total_pde_loss / num_batches, total_data_loss / num_batches) if num_batches > 0 else (0.0, 0.0, 0.0)

    

def train(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device, epochs: int, test_loader: PINNDataloader | None = None, verbose: bool = True):
    model.to(device)

    train_losses: list[float] = []
    pde_losses: list[float] = []
    data_losses: list[float] = []
    times: list[float] = []

    test_losses: list[float] | None = [] if test_loader is not None else None


    for epoch in range(epochs):
        t0 = time.time()
        trail_loss, pde_loss, data_loss = train_one_epoch(model, train_loader, optimizer, device)
        t1 = time.time()
        times.append(t1 - t0)
        train_losses.append(trail_loss)
        pde_losses.append(pde_loss)
        data_losses.append(data_loss)


        if test_losses is not None and test_loader is not None:
            test_losses.append(accuracy(model, test_loader, device))

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Data Loss: {data_losses[-1]:.4f}, PDE Loss: {pde_losses[-1]:.4f}",  end="")
            if test_losses is not None:
                print(f", Test Loss: {test_losses[-1]:.4f}", end="")
            print()

    return train_losses, pde_losses, data_losses, test_losses, times
    


#def train_one_epoch(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: str | torch.device) -> tuple[float, float, float]:
#    model.train()
#    total_loss = 0.0
#    total_pde_loss = 0.0
#    total_data_loss = 0.0
#    n_batches = 0
#
#    for batch in train_loader:
#        optimizer.zero_grad()
#        loss, pde_loss, data_loss = model.loss(batch, device)
#        loss.backward()
#        
#        # Add gradient clipping
#        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#        
#        optimizer.step()
#
#        total_loss += loss.item()
#        total_pde_loss += pde_loss.item()
#        total_data_loss += data_loss.item()
#        n_batches += 1
#
#    return total_loss / n_batches, total_pde_loss / n_batches, total_data_loss / n_batches
#
#    
#
#def train(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device, epochs: int, test_loader: PINNDataloader | None = None, verbose: bool = True):
#    model.to(device)
#
#    train_losses: list[float] = []
#    pde_losses: list[float] = []
#    data_losses: list[float] = []
#    times: list[float] = []
#
#    test_losses: list[float] | None = [] if test_loader is not None else None
#
#
#    for epoch in range(epochs):
#        t0 = time.time()
#        trail_loss, pde_loss, data_loss = train_one_epoch(model, train_loader, optimizer, device)
#        t1 = time.time()
#        times.append(t1 - t0)
#        train_losses.append(trail_loss)
#        pde_losses.append(pde_loss)
#        data_losses.append(data_loss)
#
#
#        if test_losses is not None and test_loader is not None:
#            test_losses.append(accuracy(model, test_loader, device))
#
#        if verbose:
#            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Data Loss: {data_losses[-1]:.4f}, PDE Loss: {pde_losses[-1]:.4f}",  end="")
#            if test_losses is not None:
#                print(f", Test Loss: {test_losses[-1]:.4f}", end="")
#            print()
#
#    return train_losses, pde_losses, data_losses, test_losses, times
#    