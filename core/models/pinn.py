import torch
from torch import nn
from torch import Tensor
import copy

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
        
        a_in = a.clone().requires_grad_(True)
        u_pred = self.forward(a_in)

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


    def loss_b(self, dirichlet: tuple[Tensor, Tensor], periodic: tuple[Tensor, Tensor], colloc: Tensor) -> Tensor:
        return self.dirichlet_loss(dirichlet) + self.alpha * self.periodic_loss(periodic) + self.beta * self.colloc_loss(colloc)
    
#    def loss(self, dirichlet: tuple[Tensor, Tensor], periodic: tuple[Tensor, Tensor], colloc: Tensor) -> Tensor:
#        idx_periodic = len(dirichlet[0]) - 1
#        idx_colloc = len(dirichlet[0]) + len(periodic[0]) - 1
#
#        a = torch.stack([
#            dirichlet[0],
#            periodic[0],
#            colloc
#        ], dim=1).clone().requires_grad_(True)
#
#        u = torch.stack([
#            dirichlet[1],
#            periodic[1],
#            self.forward(colloc)
#        ], dim=1).clone().requires_grad_(True)
#
#        dirichlet_loss = torch.nn.functional.mse_loss(self.forward(a[:idx_periodic, ...]), u[:idx_periodic, ...])
#        periodic_loss = torch.nn.functional.mse_loss(self.forward(a[idx_periodic:idx_colloc, ...]), self.forward(u[idx_periodic:idx_colloc, ...]))
#        colloc_loss = torch.nn.functional.mse_loss(self.pde(a[idx_colloc:, ...], u[idx_colloc:, ...]), torch.zeros_like(u[idx_colloc:, ...]))
#
#        self.dirichlet_loss_v = dirichlet_loss
#        self.periodic_loss_v = periodic_loss
#        self.pde_loss_v = colloc_loss
#        self.loss_v = dirichlet_loss + self.alpha * periodic_loss + self.beta * colloc_loss
#        return self.loss_v


#    def loss(self, dirichlet: tuple[Tensor, Tensor], periodic: tuple[Tensor, Tensor], colloc: Tensor) -> Tensor:
#        a = torch.stack([
#            dirichlet[0],
#            periodic[0],
#            periodic[1],
#            colloc
#        ], dim=1).clone().requires_grad_(True)
#
#        idx_periodic_0 = len(dirichlet[0])
#        idx_periodic_1 = idx_periodic_0 + len(periodic[0])
#        idx_colloc = idx_periodic_1 + len(colloc)
#
#        u = self.forward(a)
#
#        dirichlet_loss = torch.nn.functional.mse_loss(u[:idx_periodic_0, ...], dirichlet[1])
#        periodic_loss = torch.nn.functional.mse_loss(u[idx_periodic_0:idx_periodic_1, ...], u[idx_periodic_1:idx_colloc, ...])
#        pde_loss = torch.nn.functional.mse_loss(self.pde(a[idx_colloc:, ...], u[idx_colloc:, ...]), torch.zeros_like(u[idx_colloc:, ...]))
#
#        self.dirichlet_loss_v = dirichlet_loss
#        self.periodic_loss_v = periodic_loss
#        self.pde_loss_v = pde_loss
#        self.loss_v = dirichlet_loss + self.alpha * periodic_loss + self.beta * pde_loss
#        return self.loss_v


    def loss_a(self, dirichlet: tuple[Tensor, Tensor], periodic: tuple[Tensor, Tensor], colloc: Tensor) -> Tensor:
        a_dirichlet, u_dirichlet = dirichlet
        a_periodic_1, a_periodic_2 = periodic
        a_colloc = colloc

        tensors_to_cat = []
        if a_dirichlet.numel() > 0:
            tensors_to_cat.append(a_dirichlet)
        if a_periodic_1.numel() > 0:
            tensors_to_cat.append(a_periodic_1)
            tensors_to_cat.append(a_periodic_2)
        if a_colloc.numel() > 0:
            tensors_to_cat.append(a_colloc)

        if not tensors_to_cat:
            device = next(self.parameters()).device
            self.dirichlet_loss_v = torch.tensor(0.0, device=device)
            self.periodic_loss_v = torch.tensor(0.0, device=device)
            self.pde_loss_v = torch.tensor(0.0, device=device)
            self.loss_v = torch.tensor(0.0, device=device)
            return self.loss_v

        a_combined = torch.cat(tensors_to_cat, dim=0).clone().requires_grad_(True)
        
        u_pred_combined = self.forward(a_combined)

        dirichlet_loss = torch.tensor(0.0, device=u_pred_combined.device)
        periodic_loss = torch.tensor(0.0, device=u_pred_combined.device)
        pde_loss = torch.tensor(0.0, device=u_pred_combined.device)


        current_pos = 0
        if a_dirichlet.numel() > 0:
            n_dirichlet = a_dirichlet.shape[0]
            u_pred_dirichlet = u_pred_combined[current_pos : current_pos + n_dirichlet]
            dirichlet_loss = torch.nn.functional.mse_loss(u_pred_dirichlet, u_dirichlet)
            current_pos += n_dirichlet

        if a_periodic_1.numel() > 0:
            n_periodic = a_periodic_1.shape[0]
            u_pred_periodic_1 = u_pred_combined[current_pos : current_pos + n_periodic]
            current_pos += n_periodic
            u_pred_periodic_2 = u_pred_combined[current_pos : current_pos + n_periodic]
            current_pos += n_periodic
            periodic_loss = torch.nn.functional.mse_loss(u_pred_periodic_1, u_pred_periodic_2)

        if a_colloc.numel() > 0:
            a_colloc_slice = a_combined#[current_pos:]
            u_pred_colloc = u_pred_combined#[current_pos:]
            pde_pred = self.pde(a_colloc_slice, u_pred_colloc)
            pde_loss = torch.nn.functional.mse_loss(pde_pred, torch.zeros_like(pde_pred))


        self.dirichlet_loss_v = dirichlet_loss
        self.periodic_loss_v = periodic_loss
        self.pde_loss_v = pde_loss

        self.loss_v = self.dirichlet_loss_v + self.alpha * self.periodic_loss_v + self.beta * self.pde_loss_v
        return self.loss_v

    def loss(
    self,
    dirichlet: tuple[Tensor, Tensor],
    periodic: tuple[Tensor, Tensor],
    colloc: Tensor,
    ) -> Tensor:
        """
        L = L_dirichlet + α · L_periodic + β · L_pde
        """
        # ──────────────────────────────────────────────────────────────────────────
        # 0. Déballage et préparation
        # ──────────────────────────────────────────────────────────────────────────
        a_dir,  u_dir            = dirichlet
        a_per_1, a_per_2         = periodic
        a_col                    = colloc

        device = next(self.parameters()).device
        zero   = lambda: torch.tensor(0.0, device=device)

        # ──────────────────────────────────────────────────────────────────────────
        # 1. DIRICHLET & PÉRIODIQUE  → un seul forward
        # ──────────────────────────────────────────────────────────────────────────
        tensors_to_cat: list[Tensor] = []
        for t in (a_dir, a_per_1, a_per_2):
            if t.numel() > 0:
                tensors_to_cat.append(t)

        if tensors_to_cat:                             # au moins un bloc non vide
            a_bc   = torch.cat(tensors_to_cat, dim=0).to(device)
            u_pred = self.forward(a_bc)                # un seul passage avant
        else:
            a_bc   = None
            u_pred = None

        cur = 0
        # 1-a) Dirichlet
        if a_dir.numel() > 0:
            n = a_dir.shape[0]
            dirichlet_loss = torch.nn.functional.mse_loss(
                u_pred[cur : cur + n], u_dir.to(device)
            )
            cur += n
        else:
            dirichlet_loss = zero()

        # 1-b) Périodique
        if a_per_1.numel() > 0:
            n = a_per_1.shape[0]
            u_p1 = u_pred[cur : cur + n]
            cur += n
            u_p2 = u_pred[cur : cur + n]
            periodic_loss = torch.nn.functional.mse_loss(u_p1, u_p2)
        else:
            periodic_loss = zero()

        # ──────────────────────────────────────────────────────────────────────────
        # 2. COLLOCATION / PDE  → tensor *feuille* séparé
        # ──────────────────────────────────────────────────────────────────────────
        if a_col.numel() > 0:
            a_col_leaf  = a_col.to(device).detach().requires_grad_(True)  # feuille
            u_col_pred  = self.forward(a_col_leaf)                        # 2ᵉ forward
            pde_resid   = self.pde(a_col_leaf, u_col_pred)
            pde_loss    = torch.nn.functional.mse_loss(
                pde_resid, torch.zeros_like(pde_resid)
            )
        else:
            pde_loss = zero()

        # ──────────────────────────────────────────────────────────────────────────
        # 3. Agrégation & sortie
        # ──────────────────────────────────────────────────────────────────────────
        self.dirichlet_loss_v = dirichlet_loss
        self.periodic_loss_v  = periodic_loss
        self.pde_loss_v       = pde_loss

        self.loss_v = (
            dirichlet_loss
            + self.alpha * periodic_loss
            + self.beta  * pde_loss
        )
        return self.loss_v


    def J(self, a: Tensor, u: Tensor) -> Tensor:
        if self.J_v is not None:
            return self.J_v
        
        J_components = []
        for i in range(u.size(1)):
            grad = torch.autograd.grad(
                outputs=u[:, i],
                inputs=a,
                grad_outputs=torch.ones_like(u[:, i], device=u.device),
                create_graph=True,
                retain_graph=True
            )[0]

            if grad is None:
                grad = torch.zeros_like(a)
            
            J_components.append(grad)

        J = torch.stack(J_components, dim=1)
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


    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        
        cls = self.__class__
        new_pinn = cls.__new__(cls)
        memo[id(self)] = new_pinn
        
        for k, v in self.__dict__.items():
            if k in ['J_v', 'H_v', 'dirichlet_loss_v', 'periodic_loss_v', 'pde_loss_v', 'loss_v']:
                setattr(new_pinn, k, None)
            else:
                setattr(new_pinn, k, copy.deepcopy(v, memo))

        return new_pinn