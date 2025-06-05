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

        # potentially to delete
        self.idx_J : int | None = None
        self.idx_H : int | None = None
        self.J_v_idx : Tensor | None = None
        self.H_v_idx : Tensor | None = None

        self.J_v : Tensor | None = None
        self.H_v : Tensor | None = None




    def reinitialize(self):
        self.idx_J = None
        self.idx_H = None
        self.J_v_idx = None
        self.H_v_idx = None

        self.data_loss_v = None
        self.pde_loss_v = None
        self.loss_v = None

        self.J_v = None
        self.H_v = None

        self.a_in = None
        self.u_pred = None


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
                #if idx == 100:
                #    print(f"Data loss idx: {idx=}, {self.u_pred[:5]=}, {u[:5]=}")

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
                residue = self.pde(self.u_pred[idx:], self.a_in[idx:], idx=idx)
                if residue.numel() == 0:
                    return torch.tensor(0.0, device=self.u_pred.device, requires_grad=True)
                return torch.nn.functional.mse_loss(residue, torch.zeros_like(residue, device=self.u_pred.device))


    def loss(self, u: Tensor, idx: int | None = None):
        data_loss = self.data_loss(u, idx)
        pde_loss = self.pde_loss(idx)
        loss = data_loss + self.lmda * pde_loss

        self.data_loss_v = data_loss
        self.pde_loss_v = pde_loss
        self.loss_v = loss

        #if idx == 100:
        #    print(f"{idx=}, {data_loss=}, {pde_loss=}, {loss=}")

        return loss
    

    def J(self):
        J = [
            torch.autograd.grad(
                outputs=self.u_pred[:, i],
                inputs=self.a_in,
                grad_outputs=torch.ones_like(self.u_pred[:, i], device=self.u_pred.device),
                retain_graph=True,
                create_graph=True
            )[0] 
            for i in range(self.u_pred.size(1))
        ]
        J_v = torch.stack(J, dim=1)
        self.J_v = J_v
        return J_v


    def H(self):
        if self.u_pred is None or self.a_in is None:
            raise RuntimeError("Call forward() before computing Hessian. Ensure model has been called with input.")
        
        if self.a_in.ndim != 2:
            raise ValueError(f"self.a_in is expected to be 2D (batch_size, input_dim). Got {self.a_in.ndim}D.")

        batch_size = self.a_in.size(0)
        output_dim = self.u_pred.size(1)
        input_dim = self.a_in.size(1)

        J = self.J() 

        hessians_for_each_output_component = []
        for o_idx in range(output_dim):
            rows_of_H_o_matrix = []
            for i_idx in range(input_dim):
                J_o_i_component = J[:, o_idx, i_idx]
                
                grad_output_row = torch.autograd.grad(
                    outputs=J_o_i_component,
                    inputs=self.a_in,
                    grad_outputs=torch.ones_like(J_o_i_component, device=self.u_pred.device),
                    retain_graph=True, 
                    create_graph=True,
                    allow_unused=True 
                )[0]
                
                if grad_output_row is None:
                    grad_output_row = torch.zeros(batch_size, input_dim, 
                                                  device=self.a_in.device, dtype=self.a_in.dtype)
                
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
        if self.data_loss_v is None:
            raise ValueError("Call loss before calling get_data_loss")
        return self.data_loss_v


    def get_pde_loss(self) -> Tensor:
        if self.pde_loss_v is None:
            raise ValueError("Call loss before calling get_pde_loss")
        return self.pde_loss_v


    # calculate the jacobian of the output with respect to the input
    def J_idx(self, idx: int | None = None):
        if self.J_v_idx is not None and self.idx_J == idx:
            return self.J_v_idx
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
                create_graph=True,
                allow_unused=True
            )[0] 
            for i in range(u_pred.size(1))
        ]

        return torch.stack(J, dim=1)
    

#    # TODO: Try with self.a_in, self.u_pred as cache
#    def J(self, a_in: Tensor, u_pred: Tensor):
#        if a_in is self.a_in and u_pred is self.u_pred and self.J_v is not None:
#            return self.J_v
#
#        J = [
#            torch.autograd.grad(
#                outputs=self.u_pred[:, i],
#                inputs=self.a_in,
#                grad_outputs=torch.ones_like(self.u_pred[:, i], device=self.u_pred.device),
#                retain_graph=True,
#                create_graph=True
#            )[0] 
#            for i in range(self.u_pred.size(1))
#        ]
#        J_v = torch.stack(J, dim=1)
#        self.J_v = J_v
#        return J_v


#    def H(self, a_in: Tensor, u_pred: Tensor):
#        if a_in is self.a_in and u_pred is self.u_pred and self.H_v is not None:
#            return self.H_v
#
#        J = self.J(a_in, u_pred)
#        H = [
#            torch.autograd.grad(
#                outputs=J[:, i],
#                inputs=a_in,
#                grad_outputs=torch.ones_like(J[:, i], device=J.device),
#                retain_graph=True,
#                create_graph=True,
#                allow_unused=True
#            )[0]
#            for i in range(J.size(1))
#        ]
#        self.H_v = torch.stack(H, dim=1)
#        return self.H_v





#
#    # calculate the hessian of the output with respect to the input
#    def H_idx(self, idx: int | None = None):
#        if self.H_v_idx is not None and self.idx_H == idx:
#            return self.H_v_idx
#        self.idx_H = idx
#
#        if self.u_pred is None or self.a_in is None:
#            raise RuntimeError("Call forward() before computing loss.")
#        
#        #u_pred = self.u_pred[idx:] if idx is not None else self.u_pred
#        a_in = self.a_in[idx:] if idx is not None else self.a_in
#        
#        J = self.J(idx)
#        H = [
#            torch.autograd.grad(
#                outputs=J[:, i],
#                inputs=a_in,
#                grad_outputs=torch.ones_like(J[:, i], device=J.device),
#                retain_graph=True,
#                create_graph=True
#            )[0]
#            for i in range(J.size(1))
#        ]
#
#        return torch.stack(H, dim=1)
#    
#
#
#    def du_dt_idx(self, idx: int) -> Tensor:
#        if self.u_pred is None or self.a_in is None:
#            raise RuntimeError("Call forward() before computing du_dt.")
#        
#        J = self.J(idx)
#        return J[:, :, 0]
#    
#
#    def du_dx_idx(self, idx: int) -> Tensor:
#        if self.u_pred is None or self.a_in is None:
#            raise RuntimeError("Call forward() before computing du_dx.")
#        
#        J = self.J(idx)
#        return J[:, :, 1:]
#    
#
#
#
#    