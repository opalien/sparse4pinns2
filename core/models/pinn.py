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
                        alpha: float= 1.0, beta: float= 1.0) -> None:
        
        super().__init__() # type: ignore
        
        self.layers = nn.ModuleList(layers)
        self.activation = activation()
        self.alpha = alpha
        self.beta = beta

        self.J_v: Tensor | None = None
        self.H_v: Tensor | None = None

        self.dirichlet_loss_v : Tensor | None = None
        self.periodic_loss_v  : Tensor | None = None
        self.pde_loss_v  : Tensor | None = None
        self.loss_v      : Tensor | None = None


    def reinitialize(self):
        self.J_v = None
        self.H_v = None

        #self.dirichlet_loss_v = None
        #self.periodic_loss_v = None
        #self.pde_loss_v = None
        #self.loss_v = None


    # a : (batch_size, input_dim)""
    # returns u : (batch_size, output_dim)
    def forward(self, a: Tensor) -> Tensor:  
        self.reinitialize()      

        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        u = self.layers[-1](a)
        return u
    

    def pde(self, a: Tensor, u: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the pde method.")


    # dirichlet : (B, input_dim) 
    def dirichlet_loss(self, dirichlet: tuple[Tensor, Tensor]) -> Tensor:
        a, u = dirichlet
        if a.size(0) == 0 or u.size(0) == 0:
            return torch.tensor(0.0, device=a.device, dtype=a.dtype)
        
        u_pred = self.forward(a)

        dirichlet_loss = torch.nn.functional.mse_loss(u_pred, u)
        self.dirichlet_loss_v = dirichlet_loss
        return dirichlet_loss


    def periodic_loss(self, periodic: tuple[Tensor, Tensor]) -> Tensor:
        a_1, a_2 = periodic
        if a_1.size(0) == 0 or a_2.size(0) == 0:
            periodic_loss = torch.tensor(0.0, device=a_1.device, dtype=a_1.dtype)
            self.periodic_loss_v = periodic_loss
            return periodic_loss

        u_pred_1 = self.forward(a_1)
        u_pred_2 = self.forward(a_2)

        periodic_loss = torch.nn.functional.mse_loss(u_pred_1, u_pred_2)
        self.periodic_loss_v = periodic_loss

        return periodic_loss


    def colloc_loss(self, colloc: Tensor) -> Tensor:
        if colloc.size(0) == 0:
            colloc_loss = torch.tensor(0.0, device=colloc.device, dtype=colloc.dtype)
            self.colloc_loss_v = colloc_loss
            return colloc_loss

        a_in = colloc.clone().requires_grad_(True)
        u_pred = self.forward(a_in)
        pde_pred = self.pde(a_in, u_pred)

        pde_loss_v = torch.nn.functional.mse_loss(pde_pred, torch.zeros_like(pde_pred))
        self.pde_loss_v = pde_loss_v
        return pde_loss_v


    def loss(self, dirichlet: tuple[Tensor, Tensor], periodic: tuple[Tensor, Tensor], colloc: Tensor) -> Tensor:
        return self.dirichlet_loss(dirichlet) + self.alpha * self.periodic_loss(periodic) + self.beta * self.colloc_loss(colloc)
    

    def J(self, a: Tensor, u: Tensor) -> Tensor:
        if self.J_v is not None:
            return self.J_v
        
        J = [
            torch.autograd.grad(
                outputs=u[:, i],
                inputs=a,
                grad_outputs=torch.ones_like(u[:, i], device=u.device),
                create_graph=True,
                retain_graph=True
            )[0]
            for i in range(u.size(1))
        ]

        J = torch.stack(J, dim=1)
        self.J_v = J
        return J


    def H(self, a: Tensor, u: Tensor) -> Tensor:
        batch_size = a.size(0)
        output_dim = u.size(1)
        input_dim = a.size(1)

        J = self.J(a, u)

        hessians_for_each_output_component = []
        for o_idx in range(output_dim):
            rows_of_H_o_matrix = []
            for i_idx in range(input_dim):
                J_o_i_component = J[:, o_idx, i_idx]
                
                grad_output_row = torch.autograd.grad(
                    outputs=J_o_i_component,
                    inputs=a,
                    grad_outputs=torch.ones_like(J_o_i_component, device=u.device),
                    retain_graph=True, 
                    create_graph=True,
                    allow_unused=True 
                )[0]
                
                if grad_output_row is None:
                    grad_output_row = torch.zeros(batch_size, input_dim, 
                                                  device=a.device, dtype=a.dtype)
                
                rows_of_H_o_matrix.append(grad_output_row)

            H_o_matrix = torch.stack(rows_of_H_o_matrix, dim=1) 
            hessians_for_each_output_component.append(H_o_matrix)
            
        H_final = torch.stack(hessians_for_each_output_component, dim=1) 

        self.H_v = H_final
        return self.H_v
    

    def get_loss(self) -> Tensor:
        if self.loss_v is None:
            raise ValueError("Call loss before calling get_loss")
        return self.loss_v
    

    def get_data_loss(self) -> Tensor:
        if self.dirichlet_loss_v is None:
            raise ValueError("Call loss before calling get_data_loss")
        return self.dirichlet_loss_v
    

    def get_pde_loss(self) -> Tensor:
        if self.pde_loss_v is None:
            raise ValueError("Call loss before calling get_pde_loss")
        return self.pde_loss_v
    

    def get_periodic_loss(self) -> Tensor:
        if self.periodic_loss_v is None:
            raise ValueError("Call loss before calling get_periodic_loss")
        return self.periodic_loss_v
