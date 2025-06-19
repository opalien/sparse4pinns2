import torch
from torch import Tensor
import numpy # For visualization, pi, and other math constants
import matplotlib.pyplot as plt
from core.models.pinn import PINN

from typing import Callable


from .solver import Solver

class SchrodingerSolver(Solver):

    def __init__(self, nT: int = 100, nX: int = 100):
        self.nT = nT
        self.nX = nX


        self.t_domain = [0, numpy.pi/2]
        self.x_domain = [-5.0, 5.0]

        self.t = torch.linspace(self.t_domain[0], self.t_domain[1], nT, dtype=torch.float64)
        
        if self.nX > 0 :
            self.dx = (self.x_domain[1] - self.x_domain[0]) / self.nX
        else:
            self.dx = 0
        
        self.x = self.x_domain[0] + torch.arange(self.nX, dtype=torch.float64) * self.dx

        self.solution = torch.zeros((nT, nX, 2), dtype=torch.float64)


        if self.nX > 0: 
            self.solution[0, :, 0] = 2.0 / torch.cosh(self.x)
            self.solution[0, :, 1] = 0.0

    def _calculate_R_terms(self, u_slice: Tensor, v_slice: Tensor, dx: float) -> tuple[Tensor, Tensor]:

        u_prev = torch.roll(u_slice, shifts=1, dims=0)  # u_{i-1}
        u_next = torch.roll(u_slice, shifts=-1, dims=0) # u_{i+1}
        uxx = (u_next - 2 * u_slice + u_prev) / (dx**2)
        
        v_prev = torch.roll(v_slice, shifts=1, dims=0)
        v_next = torch.roll(v_slice, shifts=-1, dims=0)
        vxx = (v_next - 2 * v_slice + v_prev) / (dx**2)
        
        u_sq_plus_v_sq = u_slice**2 + v_slice**2
 
        
        R_u = -0.5 * vxx - u_sq_plus_v_sq * v_slice
        R_v =  0.5 * uxx + u_sq_plus_v_sq * u_slice
        
        return R_u, R_v

    def solve(self, newton_iters: int = 10, newton_tol: float = 1e-8) -> None:
        if self.nT <= 1:
            return

        dt = (self.t[1] - self.t[0]).item()
        num_spatial_points = self.nX

        if num_spatial_points == 0:
            print("Warning: nX is 0, no spatial points to solve for.")
            return
        # nX must be >=1. For 2nd derivatives, at least 3 for non-trivial interactions.
        if num_spatial_points < 3: # includes nX=1, nX=2
             print(f"Warning: nX={num_spatial_points} is very small for 2nd order derivatives with periodic BCs. Results may be inaccurate.")

        for n in range(self.nT - 1):
            if n % (max(1, (self.nT-1)//10)) == 0 or n == self.nT - 2 : # Print progress roughly 10 times
                print(f"Solving time step {n+1}/{self.nT-1}")

            u_n_full = self.solution[n, :, 0] # Real part at current time t_n
            v_n_full = self.solution[n, :, 1] # Imaginary part at current time t_n

            R_u_n, R_v_n = self._calculate_R_terms(u_n_full, v_n_full, self.dx)

            C_u = u_n_full + 0.5 * dt * R_u_n
            C_v = v_n_full + 0.5 * dt * R_v_n

            U_k_u = u_n_full.clone()
            U_k_v = v_n_full.clone()

            for iter_count in range(newton_iters):
                R_U_k_u, R_U_k_v = self._calculate_R_terms(U_k_u, U_k_v, self.dx)
                
                G_u_vec = U_k_u - 0.5 * dt * R_U_k_u - C_u
                G_v_vec = U_k_v - 0.5 * dt * R_U_k_v - C_v
                G_stacked = torch.cat((G_u_vec, G_v_vec), dim=0)

                J = torch.zeros((2 * num_spatial_points, 2 * num_spatial_points), 
                                dtype=self.solution.dtype, device=self.solution.device)

                # Derivatives of R_U_k_u = -0.5 * (U_k_v)_xx - (U_k_u^2 + U_k_v^2) * U_k_v
                # Derivatives of R_U_k_v =  0.5 * (U_k_u)_xx + (U_k_u^2 + U_k_v^2) * U_k_u

                dRuu_dUu_diag_i = -2 * U_k_u * U_k_v
                dRuu_dUv_diag_i = 1/self.dx**2 - (U_k_u**2 + 3 * U_k_v**2) # d/dUv_i (-0.5 Vxx_i) = 1/dx^2
                
                dRv_dUu_diag_i = -1/self.dx**2 + (3 * U_k_u**2 + U_k_v**2) # d/dUu_i (0.5 Uxx_i) = -1/dx^2
                dRv_dUv_diag_i = 2 * U_k_u * U_k_v
                
                const_offdiag_uv_deriv = -0.5 * (1 / self.dx**2) # d/dUv_{i-1} (-0.5 Vxx_i)
                const_offdiag_vu_deriv =  0.5 * (1 / self.dx**2) # d/dUu_{i-1} (0.5 Uxx_i)

                # J_uu block: I - 0.5*dt * dR_u/dU_u
                J_uu_diag_terms = 1.0 - 0.5 * dt * dRuu_dUu_diag_i
                J_uu_block = torch.diag(J_uu_diag_terms)
                J[:num_spatial_points, :num_spatial_points] = J_uu_block

                # J_uv block: -0.5*dt * dR_u/dU_v
                J_uv_diag_terms = -0.5 * dt * dRuu_dUv_diag_i
                J_uv_offdiag_val = -0.5 * dt * const_offdiag_uv_deriv
                J_uv_block = torch.diag(J_uv_diag_terms)
                for i in range(num_spatial_points): # Periodic BCs for spatial derivatives
                    J_uv_block[i, (i - 1 + num_spatial_points) % num_spatial_points] += J_uv_offdiag_val
                    J_uv_block[i, (i + 1) % num_spatial_points] += J_uv_offdiag_val
                J[:num_spatial_points, num_spatial_points:] = J_uv_block
                
                # J_vu block: -0.5*dt * dR_v/dU_u
                J_vu_diag_terms = -0.5 * dt * dRv_dUu_diag_i
                J_vu_offdiag_val = -0.5 * dt * const_offdiag_vu_deriv
                J_vu_block = torch.diag(J_vu_diag_terms)
                for i in range(num_spatial_points): # Periodic BCs for spatial derivatives
                    J_vu_block[i, (i - 1 + num_spatial_points) % num_spatial_points] += J_vu_offdiag_val
                    J_vu_block[i, (i + 1) % num_spatial_points] += J_vu_offdiag_val
                J[num_spatial_points:, :num_spatial_points] = J_vu_block

                # J_vv block: I - 0.5*dt * dR_v/dU_v
                J_vv_diag_terms = 1.0 - 0.5 * dt * dRv_dUv_diag_i
                J_vv_block = torch.diag(J_vv_diag_terms)
                J[num_spatial_points:, num_spatial_points:] = J_vv_block
                
                try:
                    dU_stacked = torch.linalg.solve(J, -G_stacked)
                except torch.linalg.LinAlgError as e:
                    # If G is small, we might be close enough. Otherwise, it's a problem.
                    if torch.norm(G_stacked) > 10 * newton_tol : # Check if G is significantly large
                         print(f"Warning: Jacobian singular or ill-conditioned at t_step {n+1}, iter {iter_count}. Error: {e}. Norm G: {torch.norm(G_stacked)}")
                         print("Solver may fail to converge accurately for this time step.")
                    dU_stacked = torch.zeros_like(G_stacked) # Stop Newton iteration for this step by setting dU to 0
                    break # Exit Newton loop
                
                dU_u = dU_stacked[:num_spatial_points]
                dU_v = dU_stacked[num_spatial_points:]

                U_k_u = U_k_u + dU_u
                U_k_v = U_k_v + dU_v
                
                norm_dU = torch.norm(dU_stacked)
                if norm_dU < newton_tol:
                    break
            
            self.solution[n + 1, :, 0] = U_k_u
            self.solution[n + 1, :, 1] = U_k_v
        
        print("Solver finished.")

    def func(self, a: Tensor) -> Tensor: # Adherence to Solver ABC
        if not isinstance(a, Tensor):
            raise TypeError("Input 'a' must be a PyTorch Tensor.")

        is_single_point = False
        if a.dim() == 1:
            if a.shape[0] != 2:
                raise ValueError("Single point input 'a' (1D tensor) must have shape (2,) for [t, x]")
            a_proc = a.unsqueeze(0)
            is_single_point = True
        elif a.dim() == 2:
            if a.shape[1] != 2:
                raise ValueError("Batch input 'a' (2D tensor) must have shape (num_points, 2)")
            a_proc = a
        else:
            raise ValueError("'a' must be a 1D or 2D tensor with coordinate pairs.")

        dev = self.solution.device
        a_proc = a_proc.to(dev)

        t_coords = a_proc[:, 0]
        x_coords_orig = a_proc[:, 1]

        x_min, x_max = self.x_domain[0], self.x_domain[1]
        domain_width = x_max - x_min
        # Map x_coords to the primary periodic domain [x_min, x_max)
        if domain_width > 0: # Avoid modulo by zero if x_min == x_max
            x_coords = x_min + (x_coords_orig - x_min) % domain_width
        else:
            x_coords = torch.full_like(x_coords_orig, x_min) # All points are x_min
        
        grid_t = self.t.to(dev)
        grid_x = self.x.to(dev) # These are x_min, x_min+dx, ..., x_max-dx

        idx_t_right = torch.searchsorted(grid_t, t_coords, right=True)
        idx_t0 = (idx_t_right - 1).clamp(min=0, max=self.nT - 1)
        idx_t1 = idx_t_right.clamp(min=0, max=self.nT - 1)
        
        if self.dx > 0:
            idx_x0 = torch.floor((x_coords - x_min) / self.dx).long().clamp(min=0, max=self.nX - 1)
        else: # Case for self.nX <=1, dx might be 0 or undefined
            idx_x0 = torch.zeros_like(x_coords, dtype=torch.long).clamp(min=0, max=max(0, self.nX - 1))


        interp_results_uv = torch.zeros((a_proc.shape[0], 2), dtype=self.solution.dtype, device=dev)

        for part_idx in range(2): # 0 for u (real), 1 for v (imaginary)
            # Time interpolation setup
            t0_vals = grid_t[idx_t0]
            t1_vals = grid_t[idx_t1]
            dt_ax = t1_vals - t0_vals
            alpha_t = torch.zeros_like(t_coords, device=dev)
            dt_nonzero_mask = dt_ax != 0
            alpha_t[dt_nonzero_mask] = (t_coords[dt_nonzero_mask] - t0_vals[dt_nonzero_mask]) / dt_ax[dt_nonzero_mask]
            alpha_t = alpha_t.clamp(0.0, 1.0)

            # Spatial interpolation setup
            U_grid_t0_x0 = self.solution[idx_t0, idx_x0, part_idx]
            U_grid_t1_x0 = self.solution[idx_t1, idx_x0, part_idx]
            
            idx_x0_plus_1_periodic = (idx_x0 + 1) % self.nX if self.nX > 0 else idx_x0
            U_grid_t0_x1_periodic = self.solution[idx_t0, idx_x0_plus_1_periodic, part_idx]
            U_grid_t1_x1_periodic = self.solution[idx_t1, idx_x0_plus_1_periodic, part_idx]

            alpha_x = torch.zeros_like(x_coords, device=dev)
            if self.dx > 0: # Avoid division by zero if dx is 0 (e.g., nX=1 or nX=0)
                 alpha_x = (x_coords - grid_x[idx_x0]) / self.dx
            
            alpha_x = alpha_x.clamp(0.0, 1.0)

            U_t0_interp_x = (1.0 - alpha_x) * U_grid_t0_x0 + alpha_x * U_grid_t0_x1_periodic
            U_t1_interp_x = (1.0 - alpha_x) * U_grid_t1_x0 + alpha_x * U_grid_t1_x1_periodic
            
            interp_results_uv[:, part_idx] = (1.0 - alpha_t) * U_t0_interp_x + alpha_t * U_t1_interp_x
        
        interp_results_uv = interp_results_uv.squeeze(0)
        #print(f"{interp_results_uv=}")
        return interp_results_uv

    def norm_operator(self):
        if self.nT <= 1 or self.nX == 0: # Added nX == 0 check
            return torch.tensor(0.0, dtype=self.solution.dtype), torch.tensor(0.0, dtype=self.solution.dtype)

        dt = (self.t[1] - self.t[0]).item() if self.nT > 1 else 1.0 # Prevent error if nT=1
        
        f_matrix_u = torch.zeros_like(self.solution[:,:,0])
        f_matrix_v = torch.zeros_like(self.solution[:,:,1])

        for n in range(self.nT):
            u_n = self.solution[n, :, 0]
            v_n = self.solution[n, :, 1]
            
            R_u_val, R_v_val = self._calculate_R_terms(u_n, v_n, self.dx)

            if n == 0:
                if self.nT > 1: # Forward difference
                    u_t_n = (self.solution[n + 1, :, 0] - u_n) / dt
                    v_t_n = (self.solution[n + 1, :, 1] - v_n) / dt
                else: # Single time point, derivative is ill-defined, consider residual as 0 or based on IC matching.
                    u_t_n = torch.zeros_like(u_n) # Placeholder
                    v_t_n = torch.zeros_like(v_n) # Placeholder
            elif n == self.nT - 1: # Backward difference
                u_t_n = (u_n - self.solution[n - 1, :, 0]) / dt
                v_t_n = (v_n - self.solution[n - 1, :, 1]) / dt
            else: # Central difference
                u_t_n = (self.solution[n + 1, :, 0] - self.solution[n - 1, :, 0]) / (2 * dt)
                v_t_n = (self.solution[n + 1, :, 1] - self.solution[n - 1, :, 1]) / (2 * dt)
            
            f_matrix_u[n, :] = u_t_n - R_u_val
            f_matrix_v[n, :] = v_t_n - R_v_val
        
        norm_sq_f = torch.sum(f_matrix_u**2 + f_matrix_v**2)
        abs_f_elementwise = torch.sqrt(f_matrix_u**2 + f_matrix_v**2)
        err_max = abs_f_elementwise.max()
        
        # Normalize by number of points and time steps
        return norm_sq_f / (self.nT * self.nX), err_max

    def visualize(self, component='abs', time_slices_to_plot=None):
        # Ensure there's data to plot
        if self.nX == 0 or self.nT == 0:
            print("No data to visualize (nX or nT is 0).")
            return

        T_mesh, X_mesh = torch.meshgrid(self.t, self.x, indexing='ij')
        
        u_sol = self.solution[:, :, 0].cpu().numpy()
        v_sol = self.solution[:, :, 1].cpu().numpy()

        plot_data_2d = None
        z_label_2d = ""
        title_2d = ""

        if component == 'abs':
            plot_data_2d = numpy.sqrt(u_sol**2 + v_sol**2)
            z_label_2d = '|h(t,x)|'
            title_2d = "Schrödinger Equation: Modulus |h|"
        elif component == 'real':
            plot_data_2d = u_sol
            z_label_2d = 'Re(h(t,x)) = u(t,x)'
            title_2d = "Schrödinger Equation: Real Part u"
        elif component == 'imag':
            plot_data_2d = v_sol
            z_label_2d = 'Im(h(t,x)) = v(t,x)'
            title_2d = "Schrödinger Equation: Imaginary Part v"
        else:
            raise ValueError("component must be 'abs', 'real', or 'imag'")

        # 2D Plot
        plt.figure(figsize=(10, 6))
        # T_mesh provides X-coordinates (time), X_mesh provides Y-coordinates (space)
        # plot_data_2d[i, j] is value at t_i, x_j
        plt.pcolormesh(T_mesh.cpu().numpy(), X_mesh.cpu().numpy(), plot_data_2d, 
                       shading='gouraud', cmap='viridis', vmin=plot_data_2d.min(), vmax=plot_data_2d.max())
        plt.colorbar(label=z_label_2d)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(title_2d)
        plt.show()

        # 1D Slices Plot
        if time_slices_to_plot and component == 'abs': # Only plot slices for modulus, as requested
            plt.figure(figsize=(10, 6))
            plot_found = False
            for t_target in time_slices_to_plot:
                if self.nT > 0:
                    # Find the closest time index in self.t
                    idx_t = torch.argmin(torch.abs(self.t.to(self.solution.device) - t_target)).item()
                    actual_t = self.t[idx_t].item()
                    
                    u_slice = self.solution[idx_t, :, 0]
                    v_slice = self.solution[idx_t, :, 1]
                    mod_h_slice = torch.sqrt(u_slice**2 + v_slice**2)
                    
                    plt.plot(self.x.cpu().numpy(), mod_h_slice.cpu().numpy(), label=f'|h(t={actual_t:.2f}, x)|')
                    plot_found = True
                else:
                    print(f"Cannot plot slice at t={t_target}, nT=0.")
            
            if plot_found:
                plt.xlabel('x')
                plt.ylabel('|h(t,x)|')
                plt.title(f'Modulus of h at specific time slices')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                plt.close() # Close figure if no plots were made




def schrodinger_pde(this: PINN, a: Tensor, u: Tensor) -> Tensor:
    J = this.J(a, u)
    H = this.H(a, u)

    u_real = u[:, 0]
    v_imag = u[:, 1]

    u_real_t = J[:, 0, 0]
    v_imag_t = J[:, 1, 0]

    u_real_xx = H[:, 0, 1, 1]
    v_imag_xx = H[:, 1, 1, 1]

    h_mod_sq = u_real**2 + v_imag**2

    pde_imag_residual = u_real_t + 0.5 * v_imag_xx + h_mod_sq * v_imag
    pde_real_residual = v_imag_t - 0.5 * u_real_xx - h_mod_sq * u_real

    residuals = torch.stack([pde_imag_residual, pde_real_residual], dim=1)
    
    return residuals


def schrodinger_dirichlet_generator() -> tuple[Tensor, Tensor]:
    t_coord = torch.tensor([0.0])
    x_coord = torch.empty(1).uniform_(-5, 5)
    
    a = torch.cat([t_coord, x_coord])

    u_real = 2.0 / torch.cosh(x_coord)
    u_imag = torch.tensor([0.0])
    
    u = torch.cat([u_real, u_imag])
        
    return a, u

def schrodinger_periodic_generator() -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.empty(1).uniform_(0, torch.pi / 2).item()
    a1 = torch.tensor([t, -5.0])
    a2 = torch.tensor([t, 5.0])
    return a1, a2


def schrodinger_colloc_generator() -> Tensor:
    t_coord = torch.empty(1).uniform_(0, torch.pi / 2)
    x_coord = torch.empty(1).uniform_(-5, 5)
    
    a = torch.cat([t_coord, x_coord])
    return a

def schrodinger_test_generator(func: Callable[[Tensor], Tensor]) -> tuple[Tensor, Tensor]:
    a = torch.zeros(2)
    a[0] = torch.empty(1).uniform_(0, torch.pi / 2)
    a[1] = torch.empty(1).uniform_(-5, 5)
    u = func(a)
    return a, u



def main(nT=1000, nX=500):
    import pickle
    import os

    solver = SchrodingerSolver(nT=nT, nX=nX)
    solver.solve(newton_iters=5, newton_tol=1e-6) 

    outpath = os.path.join(os.path.dirname(__file__), "schrodinger_solver.pkl")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(solver, f)
    print(f"Solver completed and saved to '{outpath}'.")
    
    print("Solver completed and saved to 'schrodinger_solver.pkl'.")