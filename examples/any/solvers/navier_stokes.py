import torch
from torch import Tensor
import numpy as np
import abc

from ..solvers.solver import Solver

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

        self.solution = torch.zeros((self.nT, self.nX, self.nY), device=self.device, dtype=torch.float64)

        if w0_initial_fn is None:
            self.w0_initial_fn = lambda x_grid, y_grid: (
                torch.sin(2 * torch.pi * x_grid) * torch.cos(2 * torch.pi * y_grid) +
                0.5 * torch.sin(4 * torch.pi * x_grid) * torch.cos(4 * torch.pi * y_grid)
            ) * 0.5
        else:
            self.w0_initial_fn = w0_initial_fn
        
        if self.nX > 0 and self.nY > 0:
            self.solution[0] = self.w0_initial_fn(self.X_grid, self.Y_grid)

        if f_forcing_fn is None:
            self.f_forcing_fn = lambda x_grid, y_grid: 0.1 * (
                torch.sin(4 * torch.pi * x_grid) + torch.cos(4 * torch.pi * y_grid)
            )
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
            if max_kx_abs > 0 : # Check to prevent division by zero if max_kx_abs is 0 (e.g. nX=1)
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
            if n > 0 :
                pass
            else: 
                pass
            
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

        a_proc = a_proc.to(self.device)
        t_coords = a_proc[:, 0]
        x_coords_orig = a_proc[:, 1]
        y_coords_orig = a_proc[:, 2]

        x_coords = x_coords_orig % 1.0
        y_coords = y_coords_orig % 1.0
        
        grid_t = self.t
        grid_x = self.x
        grid_y = self.y

        idx_t_right = torch.searchsorted(grid_t, t_coords, right=True)
        idx_t0 = (idx_t_right - 1).clamp(min=0, max=self.nT - 1)
        idx_t1 = idx_t_right.clamp(min=0, max=self.nT - 1)
        
        t0_vals = grid_t[idx_t0]
        t1_vals = grid_t[idx_t1]
        dt_ax = t1_vals - t0_vals
        alpha_t = torch.zeros_like(t_coords, device=self.device, dtype=torch.float64)
        dt_nonzero_mask = dt_ax != 0
        alpha_t[dt_nonzero_mask] = (t_coords[dt_nonzero_mask] - t0_vals[dt_nonzero_mask]) / dt_ax[dt_nonzero_mask]
        alpha_t = alpha_t.clamp(0.0, 1.0)

        idx_x0 = torch.floor(x_coords / self.dx).long().clamp(min=0, max=max(0,self.nX - 1)) if self.dx > 0 else torch.zeros_like(x_coords, dtype=torch.long)
        idx_x1_periodic = (idx_x0 + 1) % self.nX if self.nX > 0 else idx_x0
        alpha_x = (x_coords - grid_x[idx_x0]) / self.dx if self.dx > 0 and self.nX > 0 else torch.zeros_like(x_coords, device=self.device, dtype=torch.float64)
        alpha_x = alpha_x.clamp(0.0, 1.0)

        idx_y0 = torch.floor(y_coords / self.dy).long().clamp(min=0, max=max(0,self.nY - 1)) if self.dy > 0 else torch.zeros_like(y_coords, dtype=torch.long)
        idx_y1_periodic = (idx_y0 + 1) % self.nY if self.nY > 0 else idx_y0
        alpha_y = (y_coords - grid_y[idx_y0]) / self.dy if self.dy > 0 and self.nY > 0 else torch.zeros_like(y_coords, device=self.device, dtype=torch.float64)
        alpha_y = alpha_y.clamp(0.0, 1.0)
        
        if self.nX == 0 or self.nY == 0: # Handle empty solution grid case
            return torch.zeros(a_proc.shape[0], device=self.device, dtype=torch.float64)


        S_t0x0y0 = self.solution[idx_t0, idx_x0, idx_y0]
        S_t0x1y0 = self.solution[idx_t0, idx_x1_periodic, idx_y0]
        S_t0x0y1 = self.solution[idx_t0, idx_x0, idx_y1_periodic]
        S_t0x1y1 = self.solution[idx_t0, idx_x1_periodic, idx_y1_periodic]

        S_t1x0y0 = self.solution[idx_t1, idx_x0, idx_y0]
        S_t1x1y0 = self.solution[idx_t1, idx_x1_periodic, idx_y0]
        S_t1x0y1 = self.solution[idx_t1, idx_x0, idx_y1_periodic]
        S_t1x1y1 = self.solution[idx_t1, idx_x1_periodic, idx_y1_periodic]

        S_t0x0_interp_y = (1.0 - alpha_y) * S_t0x0y0 + alpha_y * S_t0x0y1
        S_t0x1_interp_y = (1.0 - alpha_y) * S_t0x1y0 + alpha_y * S_t0x1y1
        S_t1x0_interp_y = (1.0 - alpha_y) * S_t1x0y0 + alpha_y * S_t1x0y1
        S_t1x1_interp_y = (1.0 - alpha_y) * S_t1x1y0 + alpha_y * S_t1x1y1

        S_t0_interp_xy = (1.0 - alpha_x) * S_t0x0_interp_y + alpha_x * S_t0x1_interp_y
        S_t1_interp_xy = (1.0 - alpha_x) * S_t1x0_interp_y + alpha_x * S_t1x1_interp_y
        
        interp_results = (1.0 - alpha_t) * S_t0_interp_xy + alpha_t * S_t1_interp_xy
        
        if is_single_point:
            return interp_results.squeeze(0)
        else:
            return interp_results

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
            elif self.nT > 0 : # Only close if figure was created
                 plt.close()


if __name__ == "__main__":
    import pickle

    nT_sim = 1000 
    nX_sim = 512
    nY_sim = 512
    nu_sim = 1e-3
    T_final_sim = 2.0 

    solver_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = NavierStokesSolver(nT=nT_sim, nX=nX_sim, nY=nY_sim, nu=nu_sim, T_final=T_final_sim, device=solver_device)
    
    solver.solve()
    pickle.dump(solver, open("navier_stokes_solver.pkl", "wb"))
    print("Solver completed and saved to 'navier_stokes_solver.pkl'.")
