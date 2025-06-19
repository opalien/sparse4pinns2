import torch
from torch import Tensor
from core.models.pinn import PINN

from collections.abc import Callable

from ..solvers.solver import Solver

import os
import pickle


class BurgerSolver(Solver):

    def __init__(self, nT: int = 100, nX: int = 100):
        self.nT = nT
        self.nX = nX

        self.t = torch.linspace(0, 1, nT)
        self.x = torch.linspace(-1, 1, nX)

        # solution (t, x)
        self.solution = torch.zeros((nT, nX), dtype=torch.float32)

        # initial condition
        u0: Callable[[Tensor], Tensor] = lambda x: -torch.sin(torch.pi * x)

        # Boundary conditions
        u_b: Callable[[Tensor], Tensor] = lambda x: torch.zeros_like(x)

        self.solution[0, :] = u0(self.x)
        self.solution[0, 0] = 0.0
        self.solution[0, -1] = 0.0


    def _pde_rhs(self, u_slice: Tensor, dx: float, nu: float) -> Tensor:
        rhs = torch.zeros_like(u_slice)
        
        ux_interior = (u_slice[2:] - u_slice[:-2]) / (2 * dx)
        uxx_interior = (u_slice[2:] - 2 * u_slice[1:-1] + u_slice[:-2]) / (dx**2)
        u_term_interior = u_slice[1:-1]
        
        rhs[1:-1] = nu * uxx_interior - u_term_interior * ux_interior
        return rhs


    def _calculate_R(self, u_slice: Tensor, dx: float, nu: float) -> Tensor:
        u_interior = u_slice[1:-1]
        u_prev = u_slice[:-2]
        u_next = u_slice[2:]

        ux = (u_next - u_prev) / (2 * dx)
        uxx = (u_next - 2 * u_interior + u_prev) / (dx**2)
        
        return nu * uxx - u_interior * ux


    def solve(self, newton_iters: int = 5, newton_tol: float = 1e-6):
        if self.nT <= 1:
            return

        dt = (self.t[1] - self.t[0]).item()
        dx = (self.x[1] - self.x[0]).item()
        nu = 0.01 / torch.pi

        num_interior_points = self.nX - 2

        if num_interior_points <= 0:
            for n in range(self.nT - 1):
                self.solution[n + 1, 0] = 0.0
                if self.nX > 1:
                    self.solution[n + 1, -1] = 0.0
            return

        for n in range(self.nT - 1):
            if n% 100 == 0:
                print(f"Solving time step {n}/{self.nT}")
            u_n_full = self.solution[n, :]
            u_n_interior = u_n_full[1:-1]

            R_u_n = self._calculate_R(u_n_full, dx, nu)
            C_vector = u_n_interior + 0.5 * dt * R_u_n

            U_k = u_n_interior.clone() 

            for _ in range(newton_iters):
                U_k_full = torch.zeros_like(u_n_full)
                U_k_full[1:-1] = U_k

                R_U_k = self._calculate_R(U_k_full, dx, nu)
                
                G_vector = U_k - 0.5 * dt * R_U_k - C_vector

                J = torch.zeros((num_interior_points, num_interior_points), dtype=self.solution.dtype, device=self.solution.device)

                for i in range(num_interior_points):
                    val_U_k_i_plus_1 = U_k[i+1] if i < num_interior_points - 1 else 0.0
                    val_U_k_i_minus_1 = U_k[i-1] if i > 0 else 0.0
                    
                    # dR_i/dU_i = -2*nu/dx^2 - (U_{k,i+1} - U_{k,i-1}) / (2dx)
                    dR_i_dU_i = -2 * nu / (dx**2) - (val_U_k_i_plus_1 - val_U_k_i_minus_1) / (2 * dx)
                    J[i, i] = 1.0 - 0.5 * dt * dR_i_dU_i

                    if i > 0:
                        # dR_i/dU_{i-1} = nu/dx^2 + U_{k,i} / (2dx)
                        dR_i_dU_i_minus_1 = nu / (dx**2) + U_k[i] / (2 * dx)
                        J[i, i-1] = -0.5 * dt * dR_i_dU_i_minus_1
                    
                    if i < num_interior_points - 1:
                        # dR_i/dU_{i+1} = nu/dx^2 - U_{k,i} / (2dx)
                        dR_i_dU_i_plus_1 = nu / (dx**2) - U_k[i] / (2 * dx)
                        J[i, i+1] = -0.5 * dt * dR_i_dU_i_plus_1
                
                try:
                    dU = torch.linalg.solve(J, -G_vector)
                except torch.linalg.LinAlgError:
                    # Fallback if Jacobian is singular, though ideally newton_tol or iters should handle this
                    dU = torch.zeros_like(U_k) 

                U_k = U_k + dU

                if torch.norm(dU) < newton_tol:
                    break
            
            self.solution[n + 1, 1:-1] = U_k
            self.solution[n + 1, 0] = 0.0  # Apply BC
            self.solution[n + 1, -1] = 0.0 # Apply BC
        

    def func(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            raise TypeError("Input 'a' must be a PyTorch Tensor.")

        if a.dim() == 1:
            if a.shape[0] != 2:
                raise ValueError("Single point input 'a' (1D tensor) must have shape (2,) for [t, x]")
            a_proc = a.unsqueeze(0)
        elif a.dim() == 2:
            if a.shape[1] != 2:
                raise ValueError("Batch input 'a' (2D tensor) must have shape (num_points, 2)")
            a_proc = a
        else:
            raise ValueError("'a' must be a 1D or 2D tensor with coordinate pairs.")

        dev = self.solution.device
        a_proc = a_proc.to(dev)

        t_coords = a_proc[:, 0]
        x_coords = a_proc[:, 1]
        
        grid_t = self.t.to(dev)
        grid_x = self.x.to(dev)

        idx_t_right = torch.searchsorted(grid_t, t_coords, right=True)
        idx_t0 = (idx_t_right - 1).clamp(min=0, max=self.nT - 1)
        idx_t1 = idx_t_right.clamp(min=0, max=self.nT - 1)

        idx_x_right = torch.searchsorted(grid_x, x_coords, right=True)
        idx_x0 = (idx_x_right - 1).clamp(min=0, max=self.nX - 1)
        idx_x1 = idx_x_right.clamp(min=0, max=self.nX - 1)
        
        t0_vals = grid_t[idx_t0]
        t1_vals = grid_t[idx_t1]
        x0_vals = grid_x[idx_x0]
        x1_vals = grid_x[idx_x1]

        U00 = self.solution[idx_t0, idx_x0]
        U01 = self.solution[idx_t0, idx_x1]
        U10 = self.solution[idx_t1, idx_x0]
        U11 = self.solution[idx_t1, idx_x1]
        
        dt_ax = t1_vals - t0_vals
        dx_ax = x1_vals - x0_vals

        alpha_t = torch.zeros_like(t_coords, device=dev)
        alpha_x = torch.zeros_like(x_coords, device=dev)

        dt_nonzero_mask = dt_ax != 0
        if dt_nonzero_mask.any():
             alpha_t[dt_nonzero_mask] = (t_coords[dt_nonzero_mask] - t0_vals[dt_nonzero_mask]) / dt_ax[dt_nonzero_mask]
        
        dx_nonzero_mask = dx_ax != 0
        if dx_nonzero_mask.any():
            alpha_x[dx_nonzero_mask] = (x_coords[dx_nonzero_mask] - x0_vals[dx_nonzero_mask]) / dx_ax[dx_nonzero_mask]

        alpha_t = alpha_t.clamp(0.0, 1.0)
        alpha_x = alpha_x.clamp(0.0, 1.0)

        U_t0_interp_x = (1.0 - alpha_x) * U00 + alpha_x * U01
        U_t1_interp_x = (1.0 - alpha_x) * U10 + alpha_x * U11

        interp_results = (1.0 - alpha_t) * U_t0_interp_x + alpha_t * U_t1_interp_x
        
        return interp_results


    def norm_operator(self):
        dx = (self.x[1] - self.x[0]).item()
        nu = 0.01 / torch.pi

        f_matrix = torch.zeros_like(self.solution)

        for n in range(self.nT):
            u_n = self.solution[n, :]
            f_n = self._pde_rhs(u_n, dx, nu)
            f_matrix[n, 1:-1] = f_n[1:-1]

        err_max = f_matrix.abs().max()

        return (torch.norm(f_matrix, p=2)**2)/ (self.nT * self.nX), err_max 


    def visualize(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        T, X = torch.meshgrid(self.t, self.x, indexing='ij')
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T.numpy(), X.numpy(), self.solution.numpy(), cmap='viridis')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u(t,x)')
        plt.title("Burgers' Equation Solution")
        plt.show()



def burger_pde(this: PINN, a: Tensor, u: Tensor) -> Tensor:

    J = this.J(a, u)
    H = this.H(a, u)

    du_dt = J[:, 0, 0]
    du_dx = J[:, 0, 1]
    du_dxx = H[:, 0, 1, 1]
    u_ = u[:, 0]


    print(f"{torch.norm(du_dt, p=2)=}, {torch.norm(du_dx, p=2)=}, {torch.norm(du_dxx, p=2)=}, {torch.norm(u_, p=2)=}")

    return du_dt + u_*du_dx - du_dxx*(0.01 / torch.pi)


def burger_dirichlet_generator() -> tuple[Tensor, Tensor]:
    a, u = None, None
    match torch.randint(0, 3, (1,)).item():
        case 0:
            a = torch.zeros(2)
            a[1] = torch.empty(1).uniform_(-1, 1)
            u = -torch.sin(torch.pi * a[1]).unsqueeze(0)

        case 1:
            a = torch.zeros(2)
            a[0] = torch.empty(1).uniform_(0, 1)
            a[1] = -1
            u = torch.zeros(1)
        
        case 2:
            a = torch.zeros(2)
            a[0] = torch.empty(1).uniform_(0, 1)
            a[1] = 1
            u = torch.zeros(1)

        case _:
            raise ValueError("Unexpected case in dirichlet_generator")
        
    return a, u


def burger_colloc_generator() -> Tensor:
    a = torch.zeros(2)
    a[0] = torch.empty(1).uniform_(0, 1)
    a[1] = torch.empty(1).uniform_(-1, 1)
    return a


def burger_test_generator(func: Callable[[Tensor], Tensor]) -> tuple[Tensor, Tensor]:
    a = torch.zeros(2)
    a[0] = torch.empty(1).uniform_(0, 1)
    a[1] = torch.empty(1).uniform_(-1, 1)
    u = func(a)
    return a, u


def main(nT: int = 10_000, nX: int = 1000):
    solver = BurgerSolver(nT=nT, nX=nX)
    solver.solve(newton_iters=5, newton_tol=1e-6)

    outpath = os.path.join(os.path.dirname(__file__), "burger_solver.pkl")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(solver, f)
    print(f"Solver completed and saved to '{outpath}'.")



# USAGE :
# python -c "from examples.any.solvers.burger import main; main()"
