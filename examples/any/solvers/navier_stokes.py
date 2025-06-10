import torch
from torch import Tensor
import numpy as np
import abc
from core.models.pinn import PINN

from ..solvers.solver import Solver

def _default_w0_initial_fn(x_grid: Tensor, y_grid: Tensor) -> Tensor:
    """
    Default initial vorticity w0(x,y).
    """
    return (
        torch.sin(2 * torch.pi * x_grid) * torch.cos(2 * torch.pi * y_grid)
        + 0.5 * torch.sin(4 * torch.pi * x_grid) * torch.cos(4 * torch.pi * y_grid)
    ) * 0.5

def _default_f_forcing_fn(x_grid: Tensor, y_grid: Tensor) -> Tensor:
    """
    Default forcing term f(x,y).
    """
    return 0.1 * (torch.sin(4 * torch.pi * x_grid) + torch.cos(4 * torch.pi * y_grid))


class NavierStokesSolver(Solver):

    def __init__(self, nT: int, nX: int, nY: int, nu: float, T_final: float, 
                 f_forcing_fn=None, w0_initial_fn=None, device=None):
        self.nT = nT
        self.nX = nX
        self.nY = nY
        self.nu = nu
        self.T_final = T_final

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.t = torch.linspace(0, self.T_final, self.nT, device=self.device, dtype=torch.float64)
        
        _x_coords = torch.linspace(0, 1, self.nX + 1, device=self.device, dtype=torch.float64)
        self.x = _x_coords[:-1]
        _y_coords = torch.linspace(0, 1, self.nY + 1, device=self.device, dtype=torch.float64)
        self.y = _y_coords[:-1]
        
        self.dx = 1.0 / self.nX if self.nX > 0 else 1.0
        self.dy = 1.0 / self.nY if self.nY > 0 else 1.0

        self.X_grid, self.Y_grid = torch.meshgrid(self.x, self.y, indexing='ij')

        # La solution ne stocke que la vorticité `w`. La fonction de courant `psi` sera calculée à la volée.
        self.solution = torch.zeros((self.nT, self.nX, self.nY), device=self.device, dtype=torch.float64)

        if w0_initial_fn is None:
            self.w0_initial_fn = _default_w0_initial_fn
        else:
            self.w0_initial_fn = w0_initial_fn
        
        if self.nX > 0 and self.nY > 0:
            self.solution[0] = self.w0_initial_fn(self.X_grid, self.Y_grid)

        if f_forcing_fn is None:
            self.f_forcing_fn = _default_f_forcing_fn
        else:
            self.f_forcing_fn = f_forcing_fn

        if self.nX > 0 and self.nY > 0:
            self.f_forcing_phys = self.f_forcing_fn(self.X_grid, self.Y_grid)
            self.f_forcing_hat = torch.fft.fft2(self.f_forcing_phys)
        else:
            self.f_forcing_phys = torch.zeros((self.nX, self.nY), device=self.device, dtype=torch.float64)
            self.f_forcing_hat = torch.zeros((self.nX, self.nY), device=self.device, dtype=torch.complex128)


        kx_vals = 2 * torch.pi * torch.fft.fftfreq(self.nX, d=self.dx, device=self.device, dtype=torch.float64)
        ky_vals = 2 * torch.pi * torch.fft.fftfreq(self.nY, d=self.dy, device=self.device, dtype=torch.float64)
        self.Kx, self.Ky = torch.meshgrid(kx_vals, ky_vals, indexing='ij')
        
        self.K_sq = self.Kx**2 + self.Ky**2
        self.K_sq_no_zero = self.K_sq.clone()
        if self.nX > 0 and self.nY > 0:
            self.K_sq_no_zero[0,0] = 1.0

        self.dealias_mask = torch.ones_like(self.K_sq, dtype=torch.bool)
        if self.nX > 0 and self.nY > 0 :
            max_kx_abs = torch.max(torch.abs(self.Kx)) if self.nX > 0 else 0
            max_ky_abs = torch.max(torch.abs(self.Ky)) if self.nY > 0 else 0
            if max_kx_abs > 0 :
                 self.dealias_mask[torch.abs(self.Kx) > (2.0/3.0) * max_kx_abs] = 0
            if max_ky_abs > 0 :
                 self.dealias_mask[torch.abs(self.Ky) > (2.0/3.0) * max_ky_abs] = 0


    def _compute_nonlinear_term_hat(self, w_hat_curr: Tensor) -> Tensor:
        if self.nX == 0 or self.nY == 0:
            return torch.zeros_like(w_hat_curr)

        psi_hat = -w_hat_curr / self.K_sq_no_zero
        psi_hat[0,0] = 0.0 

        ux_hat = 1j * self.Ky * psi_hat
        uy_hat = -1j * self.Kx * psi_hat

        wx_hat = 1j * self.Kx * w_hat_curr
        wy_hat = 1j * self.Ky * w_hat_curr

        ux_phys = torch.fft.ifft2(ux_hat).real
        uy_phys = torch.fft.ifft2(uy_hat).real
        wx_phys = torch.fft.ifft2(wx_hat).real
        wy_phys = torch.fft.ifft2(wy_hat).real
        
        nonlinear_term_phys = ux_phys * wx_phys + uy_phys * wy_phys
        nonlinear_term_hat = torch.fft.fft2(nonlinear_term_phys)
        nonlinear_term_hat *= self.dealias_mask
        return nonlinear_term_hat


    def solve(self) -> None:
        if self.nT <= 1 or self.nX == 0 or self.nY == 0:
            return

        dt = (self.t[1] - self.t[0]).item()
        w_hat_n = torch.fft.fft2(self.solution[0])
        
        N_hat_n = self._compute_nonlinear_term_hat(w_hat_n)
        N_hat_nm1 = N_hat_n.clone() 
        
        linear_op_implicit = 1.0 + 0.5 * dt * self.nu * self.K_sq
        linear_op_explicit = 1.0 - 0.5 * dt * self.nu * self.K_sq

        for n in range(self.nT - 1):
            if n % 10 == 0:
                print(f"Solving time step {n}/{self.nT}")
            
            if n == 0: 
                rhs_advection_forced = -dt * N_hat_n + dt * self.f_forcing_hat
            else: 
                rhs_advection_forced = -dt * (1.5 * N_hat_n - 0.5 * N_hat_nm1) + dt * self.f_forcing_hat

            w_hat_np1 = (linear_op_explicit * w_hat_n + rhs_advection_forced) / linear_op_implicit
            
            if self.f_forcing_hat[0,0].abs() < 1e-15 and self.solution[0].mean().abs() < 1e-15 :
                w_hat_np1[0,0] = 0.0 

            self.solution[n+1] = torch.fft.ifft2(w_hat_np1).real
            
            N_hat_nm1 = N_hat_n 
            w_hat_n = w_hat_np1
            if n < self.nT - 2: 
                 N_hat_n = self._compute_nonlinear_term_hat(w_hat_n)

    def _get_psi_from_w(self, w_grid: Tensor) -> Tensor:
        """Calcule la fonction de courant psi à partir de la vorticité w sur la grille."""
        if self.nX == 0 or self.nY == 0:
            return torch.zeros_like(w_grid)
        
        w_hat = torch.fft.fft2(w_grid)
        psi_hat = -w_hat / self.K_sq_no_zero
        # Le mode (0,0) correspond à la moyenne de psi, que l'on peut fixer à 0.
        psi_hat[..., 0, 0] = 0.0
        
        psi_grid = torch.fft.ifft2(psi_hat).real
        return psi_grid

    def func(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            raise TypeError("Input 'a' must be a PyTorch Tensor.")

        is_single_point = False
        if a.dim() == 1:
            if a.shape[0] != 3:
                raise ValueError("Single point input 'a' (1D tensor) must have shape (3,) for [t, x, y]")
            a_proc = a.unsqueeze(0)
            is_single_point = True
        elif a.dim() == 2:
            if a.shape[1] != 3:
                raise ValueError("Batch input 'a' (2D tensor) must have shape (num_points, 3)")
            a_proc = a
        else:
            raise ValueError("'a' must be a 1D or 2D tensor with coordinate triplets.")
        
        num_points = a_proc.shape[0]
        a_proc = a_proc.to(self.device)
        t_coords = a_proc[:, 0]
        x_coords_orig = a_proc[:, 1]
        y_coords_orig = a_proc[:, 2]

        x_coords = x_coords_orig % 1.0
        y_coords = y_coords_orig % 1.0
        
        grid_t, grid_x, grid_y = self.t, self.x, self.y

        # Indices et coefficients pour l'interpolation temporelle
        idx_t_right = torch.searchsorted(grid_t, t_coords, right=True)
        idx_t0 = (idx_t_right - 1).clamp(min=0, max=self.nT - 1)
        idx_t1 = idx_t_right.clamp(min=0, max=self.nT - 1)
        
        t0_vals = grid_t[idx_t0]
        t1_vals = grid_t[idx_t1]
        dt_ax = t1_vals - t0_vals
        alpha_t = torch.zeros_like(t_coords, device=self.device, dtype=torch.float64)
        dt_nonzero_mask = dt_ax != 0
        alpha_t[dt_nonzero_mask] = (t_coords[dt_nonzero_mask] - t0_vals[dt_nonzero_mask]) / dt_ax[dt_nonzero_mask]
        alpha_t = alpha_t.clamp(0.0, 1.0).unsqueeze(-1) # Pour le broadcasting (B, 1)

        # Indices et coefficients pour l'interpolation spatiale
        idx_x0 = torch.floor(x_coords / self.dx).long().clamp(min=0, max=max(0,self.nX - 1)) if self.dx > 0 else torch.zeros_like(x_coords, dtype=torch.long)
        idx_x1_periodic = (idx_x0 + 1) % self.nX if self.nX > 0 else idx_x0
        alpha_x = ((x_coords - grid_x[idx_x0]) / self.dx if self.dx > 0 and self.nX > 0 else torch.zeros_like(x_coords, device=self.device, dtype=torch.float64)).clamp(0.0, 1.0)

        idx_y0 = torch.floor(y_coords / self.dy).long().clamp(min=0, max=max(0,self.nY - 1)) if self.dy > 0 else torch.zeros_like(y_coords, dtype=torch.long)
        idx_y1_periodic = (idx_y0 + 1) % self.nY if self.nY > 0 else idx_y0
        alpha_y = ((y_coords - grid_y[idx_y0]) / self.dy if self.dy > 0 and self.nY > 0 else torch.zeros_like(y_coords, device=self.device, dtype=torch.float64)).clamp(0.0, 1.0)
        
        # Cas où la grille est vide
        if self.nX == 0 or self.nY == 0:
            return torch.zeros((num_points, 2), device=self.device, dtype=torch.float64)

        # Extraction des grilles de vorticité aux temps t0 et t1
        w_grid_t0 = self.solution[idx_t0]
        w_grid_t1 = self.solution[idx_t1]

        # Calcul des grilles de fonction de courant correspondantes
        psi_grid_t0 = self._get_psi_from_w(w_grid_t0)
        psi_grid_t1 = self._get_psi_from_w(w_grid_t1)
        
        # Fonction d'interpolation pour un champ donné (w ou psi)
        def _interpolate_field(field_grid_t0, field_grid_t1):
            # Indexation pour récupérer les valeurs aux 8 coins du cube d'interpolation pour chaque point du batch
            batch_indices = torch.arange(num_points, device=self.device)
            
            f_t0x0y0 = field_grid_t0[batch_indices, idx_x0, idx_y0]
            f_t0x1y0 = field_grid_t0[batch_indices, idx_x1_periodic, idx_y0]
            f_t0x0y1 = field_grid_t0[batch_indices, idx_x0, idx_y1_periodic]
            f_t0x1y1 = field_grid_t0[batch_indices, idx_x1_periodic, idx_y1_periodic]

            f_t1x0y0 = field_grid_t1[batch_indices, idx_x0, idx_y0]
            f_t1x1y0 = field_grid_t1[batch_indices, idx_x1_periodic, idx_y0]
            f_t1x0y1 = field_grid_t1[batch_indices, idx_x0, idx_y1_periodic]
            f_t1x1y1 = field_grid_t1[batch_indices, idx_x1_periodic, idx_y1_periodic]

            # Interpolation bilinéaire sur le plan xy au temps t0
            f_t0_interp_y0 = (1.0 - alpha_x) * f_t0x0y0 + alpha_x * f_t0x1y0
            f_t0_interp_y1 = (1.0 - alpha_x) * f_t0x0y1 + alpha_x * f_t0x1y1
            f_t0_interp_xy = (1.0 - alpha_y) * f_t0_interp_y0 + alpha_y * f_t0_interp_y1

            # Interpolation bilinéaire sur le plan xy au temps t1
            f_t1_interp_y0 = (1.0 - alpha_x) * f_t1x0y0 + alpha_x * f_t1x1y0
            f_t1_interp_y1 = (1.0 - alpha_x) * f_t1x0y1 + alpha_x * f_t1x1y1
            f_t1_interp_xy = (1.0 - alpha_y) * f_t1_interp_y0 + alpha_y * f_t1_interp_y1
            
            # Interpolation linéaire finale en temps
            interp_results = (1.0 - alpha_t.squeeze(-1)) * f_t0_interp_xy + alpha_t.squeeze(-1) * f_t1_interp_xy
            return interp_results

        # Interpolation pour la vorticité (w) et la fonction de courant (psi)
        w_interp = _interpolate_field(w_grid_t0, w_grid_t1)
        psi_interp = _interpolate_field(psi_grid_t0, psi_grid_t1)

        # Empile les résultats pour obtenir une sortie de dimension 2
        final_results = torch.stack([psi_interp, w_interp], dim=1)
        
        return final_results.squeeze(0) if is_single_point else final_results

    def visualize(self, time_point_idx=-1, time_slices_to_plot=None):
        import matplotlib.pyplot as plt

        if self.nX == 0 or self.nY == 0 or self.nT == 0:
            print("No data to visualize (nX, nY, or nT is 0).")
            return

        w_to_plot = self.solution[time_point_idx].cpu().numpy()
        t_val = self.t[time_point_idx].item()

        plt.figure(figsize=(8, 6))
        plt.contourf(self.x.cpu().numpy(), self.y.cpu().numpy(), w_to_plot.T, levels=50, cmap='viridis')
        plt.colorbar(label=f'$w(x,y,t={t_val:.2f})$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vorticity field $w$ at $t={t_val:.2f}$')
        plt.axis('scaled')
        plt.show()

        if time_slices_to_plot:
            if self.nY == 0 :
                print("Cannot plot 1D slices as nY is 0.")
                return

            y_slice_idx = self.nY // 2
            y_val_slice = self.y[y_slice_idx].item()

            plt.figure(figsize=(10,6))
            plot_found = False
            for t_target in time_slices_to_plot:
                if self.nT > 0:
                    idx_t = torch.argmin(torch.abs(self.t.to(self.device) - t_target)).item()
                    actual_t = self.t[idx_t].item()
                    
                    w_slice_data = self.solution[idx_t, :, y_slice_idx].cpu().numpy()
                    
                    plt.plot(self.x.cpu().numpy(), w_slice_data, label=f'$w(x, y={y_val_slice:.2f}, t={actual_t:.2f})$')
                    plot_found = True
                else:
                    print(f"Cannot plot slice at t={t_target}, nT=0.")
            
            if plot_found:
                plt.xlabel('x')
                plt.ylabel(f'$w(x, y={y_val_slice:.2f}, t)$')
                plt.title(f'Vorticity slices along x (at $y={y_val_slice:.2f}$)')
                plt.legend()
                plt.grid(True)
                plt.show()
            elif self.nT > 0 :
                 plt.close()


def navier_stokes_pde(this: PINN, a: Tensor, u: Tensor) -> Tensor:
    w_pred = u[:, 1]
    
    J = this.J(a, u)
    H = this.H(a, u)

    dpsi_dx = J[:, 0, 1]
    dpsi_dy = J[:, 0, 2]
    d2psi_dx2 = H[:, 0, 1, 1]
    d2psi_dy2 = H[:, 0, 2, 2]

    dw_dt = J[:, 1, 0]
    dw_dx = J[:, 1, 1]
    dw_dy = J[:, 1, 2]
    d2w_dx2 = H[:, 1, 1, 1]
    d2w_dy2 = H[:, 1, 2, 2]

    nu = 1e-3
    x_colloc = a[:, 1]
    y_colloc = a[:, 2]

    laplacian_psi = d2psi_dx2 + d2psi_dy2
    residual_poisson = w_pred + laplacian_psi

    f_forcing = 0.1 * (torch.sin(4 * torch.pi * x_colloc) + torch.cos(4 * torch.pi * y_colloc))
    u_velocity = dpsi_dy
    v_velocity = -dpsi_dx
    advection_term = u_velocity * dw_dx + v_velocity * dw_dy
    diffusion_term = nu * (d2w_dx2 + d2w_dy2)
    residual_transport = dw_dt + advection_term - diffusion_term - f_forcing

    residuals = torch.stack([residual_poisson, residual_transport], dim=1)
    return residuals


def navier_stokes_dirichlet_generator() -> tuple[Tensor, Tensor]:
    t = torch.tensor([0.0])
    x = torch.empty(1).uniform_(0, 1)
    y = torch.empty(1).uniform_(0, 1)
    
    a = torch.cat([t, x, y])

    w_val = _default_w0_initial_fn(x, y)
    psi_val = torch.tensor([0.0]) 

    u = torch.cat([psi_val, w_val])
    return a, u


def navier_stokes_periodic_generator() -> tuple[Tensor, Tensor]:
    T_final = 2.0
    t = torch.empty(1).uniform_(0, T_final)

    if torch.rand(1).item() > 0.5:
        y = torch.empty(1).uniform_(0, 1)
        a1 = torch.cat([t, torch.tensor([0.0]), y])
        a2 = torch.cat([t, torch.tensor([1.0]), y])
    else:
        x = torch.empty(1).uniform_(0, 1)
        a1 = torch.cat([t, x, torch.tensor([0.0])])
        a2 = torch.cat([t, x, torch.tensor([1.0])])
    
    return a1, a2


def navier_stokes_colloc_generator() -> Tensor:
    T_final = 2.0
    t = torch.empty(1).uniform_(0, T_final)
    x = torch.empty(1).uniform_(0, 1)
    y = torch.empty(1).uniform_(0, 1)
    
    a = torch.cat([t, x, y])
    return a


def main():
    import os
    import pickle

    nT_sim = 1000 
    nX_sim = 128
    nY_sim = 128
    nu_sim = 1e-3
    T_final_sim = 2.0 

    solver = NavierStokesSolver(nT=nT_sim, nX=nX_sim, nY=nY_sim, nu=nu_sim, T_final=T_final_sim)    
    solver.solve()

    outpath = os.path.join(os.path.dirname(__file__), "navier_stokes_solver.pkl")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(solver, f)
    print(f"Solver completed and saved to '{outpath}'.")