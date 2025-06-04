from ..tensors.permutation import PermutationTensor
from torch import Tensor, nn
import torch

from einops import rearrange, einsum # type: ignore

from scipy.optimize import linear_sum_assignment # type: ignore

from torch.autograd.function import FunctionCtx

from ..layers.parameterlike import ParameterLike

from ..tensors.tensorlike import TensorLike



class PermutationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, X: Tensor) -> Tensor:
        device = X.device
        dtype = X.dtype
        n = X.shape[0]

        with torch.no_grad():
            X_np = -X.detach().cpu().numpy() # type: ignore
            row_ind, col_ind = linear_sum_assignment(X_np) # type: ignore
            P_opt = torch.zeros((n, n), dtype=dtype, device=device)
            P_opt[row_ind, col_ind] = 1.0

        return P_opt
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor): # type: ignore
        return grad_output


class LearnablePermutation(ParameterLike):
    def __init__(self, Xinit: Tensor | PermutationTensor):
        super().__init__() # type: ignore
        self.X = nn.Parameter(Xinit.dense if isinstance(Xinit, PermutationTensor) else Xinit)
        self.n = self.X.shape[0]
        self.P: Tensor | None = None
        self.Pvec: PermutationTensor | None = None


    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] != self.n:
            raise ValueError(f"Input tensor must have shape {self.n}, but got {x.shape[-1]}")
         
        self.P = PermutationSTE.apply(self.X) # type: ignore

        return einsum(self.P, x, "m n, n ... -> m ...") # type: ignore


    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        match other:
            case TensorLike():
                return self.__matmul__(other.dense)
            
            case Tensor():
                return self.forward(other)
            
            case _:
                return NotImplemented


    @property
    def dense(self) -> Tensor:
        return PermutationSTE.apply(self.X) # type: ignore


    def tensor(self) -> Tensor:
        return PermutationTensor.from_dense(self.P) # type: ignore