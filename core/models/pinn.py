import torch
from torch import nn
from torch import Tensor

from collections.abc import Iterable


from ..layers.monarch import MonarchLinear
from ..layers.steam import STEAMLinear

DLinear = nn.Linear | MonarchLinear | STEAMLinear


class PINN(nn.Module):

    def __init__(self,  layers: Iterable[DLinear],
                        activation: type[nn.Module]=nn.Tanh, 
                        lmda:float =1.0) -> None:
        
        super().__init__() # type: ignore
        
        self.layers = nn.ModuleList(layers)
        self.activation = activation()
        self.lmda = lmda

        self.a_in   : Tensor | None = None
        self.u_pred : Tensor | None = None

        self.data_loss_v : Tensor | None = None
        self.pde_loss_v  : Tensor | None = None
        self.loss_v      : Tensor | None = None

        self.idx_J : int | None = None
        self.idx_H : int | None = None
        self.J_v : Tensor | None = None
        self.H_v : Tensor | None = None


    def reinitialize(self):
        self.idx_J = None
        self.idx_H = None
        self.J_v = None
        self.H_v = None


    # a : (batch_size, input_dim)
    # returns u : (batch_size, output_dim)
    def forward(self, a: Tensor) -> Tensor:
        self.reinitialize()

        a = a.clone().requires_grad_(True)
        self.a_in = a
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        u = self.layers[-1](a)
        self.u_pred = u
        return u
    

    def data_loss(self, u: Tensor, idx: int | None):
        if self.u_pred is None:
            raise RuntimeError("Call forward() before computing data loss.")
        
        idx = self.u_pred.size(0) if idx is None else idx

        match idx:
            case 0:
                return torch.tensor(0.0, device=self.u_pred.device)

            case _:
                return torch.nn.functional.mse_loss(
                    self.u_pred[:idx], u[:idx]
                )


    def pde(self, u_pred: Tensor, a_in: Tensor, idx: int | None = None) -> Tensor:
        raise NotImplementedError("Subclasses must implement the pde method.")


    def pde_loss(self, idx: int | None):
        if self.a_in is None or self.u_pred is None:
            raise RuntimeError("Call forward() before computing PDE loss.")
        
        idx = self.a_in.size(0) if idx is None else idx

        match idx:
            case _ if idx == self.a_in.size(0):
                return torch.tensor(0.0, device=self.a_in.device)
            
            case _:
                residue = self.pde(self.u_pred, self.a_in, idx)
                if residue.numel() == 0:
                    return torch.tensor(0.0, device=self.u_pred.device, requires_grad=True)
                return torch.nn.functional.mse_loss(residue, torch.zeros_like(residue, device=self.u_pred.device))


    def loss(self, u: Tensor, idx: int | None = None):
        data_loss = self.data_loss(u, idx)
        pde_loss = self.pde_loss(idx)
        return data_loss + self.lmda * pde_loss
    

    # calculate the jacobian of the output with respect to the input
    def J(self, idx: int | None = None):
        if self.J_v is not None and self.idx_J == idx:
            return self.J_v
        self.idx_J = idx

        if self.u_pred is None or self.a_in is None:
            raise RuntimeError("Call forward() before computing loss.")
        
        u_pred = self.u_pred[idx:] if idx is not None else self.u_pred
        a_in = self.a_in[idx:] if idx is not None else self.a_in

        J = [
            torch.autograd.grad(
                outputs=u_pred[:, i],
                inputs=a_in,
                grad_outputs=torch.ones_like(u_pred[:, i], device=u_pred.device),
                retain_graph=True,
                create_graph=True
            )[0] 
            for i in range(u_pred.size(1))
        ]

        return torch.stack(J, dim=1)


    # calculate the hessian of the output with respect to the input
    def H(self, idx: int | None = None):
        if self.H_v is not None and self.idx_H == idx:
            return self.H_v
        self.idx_H = idx

        if self.u_pred is None or self.a_in is None:
            raise RuntimeError("Call forward() before computing loss.")
        
        #u_pred = self.u_pred[idx:] if idx is not None else self.u_pred
        a_in = self.a_in[idx:] if idx is not None else self.a_in
        
        J = self.J(idx)
        H = [
            torch.autograd.grad(
                outputs=J[:, i],
                inputs=a_in,
                grad_outputs=torch.ones_like(J[:, i], device=J.device),
                retain_graph=True,
                create_graph=True
            )[0]
            for i in range(J.size(1))
        ]

        return torch.stack(H, dim=1)
    

