from __future__ import annotations
from ..tensors.tensorlike import TensorLike
from ..tensors.block_diag import BlockDiagTensor
from ..tensors.permutation import bit_rev, BitRevPermutationTensor

from torch import Tensor, nn
import torch
import copy

from ..utils.butterfly import blockdiag_butterfly_project


from  einops import rearrange, einsum # type: ignore


class MonarchTensor(TensorLike):
    def __new__(cls, tensor: Tensor, *args, **kwargs): # type: ignore
        if not isinstance(tensor, Tensor): # type: ignore
            raise TypeError("MonarchTensor must be initialized with a Tensor")
        if tensor.ndim != 4:
            raise ValueError("MonarchTensor must have 4 dimensions")
        
        if tensor.shape[1] != tensor.shape[2] != tensor.shape[3]:
            raise ValueError("MonarchTensor must have 4 dimensions with the second and third dimensions equal")

        return tensor.as_subclass(cls)


    def __init__(self, tensor: Tensor, *args, **kwargs): # type: ignore
        self.m: int = tensor.shape[1]
        self.n: int = self.m**2
        self.P1 = BitRevPermutationTensor(self.n)
        self.P2  = BitRevPermutationTensor(self.n)


    @property 
    def L(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[1, ...])


    @property 
    def R(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[0, ...])


    @L.setter
    def L(self, new_L: BlockDiagTensor):
        if new_L.shape != (self.m, self.m, self.m):
            raise ValueError(f"Dimension mismatch: L must have shape ({self.m}, {self.m}, {self.m}), but got {new_L.shape}")
        self[1, ...] = new_L.clone().detach().data


    @R.setter
    def R(self, new_R: BlockDiagTensor):
        if new_R.shape != (self.m, self.m, self.m):
            raise ValueError(f"Dimension mismatch: R must have shape ({self.m}, {self.m}, {self.m}), but got {new_R.shape}")
        self[0, ...] = new_R.clone().detach().data


    @property
    def dense(self) -> Tensor:
        return self.P2 @ self.L @ self.P1 @ self.R
    
    @staticmethod
    def from_dense(tensor: Tensor) -> MonarchTensor:
        R, L = blockdiag_butterfly_project(tensor)
        return MonarchTensor( torch.stack([R, L]) )


    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return MonarchTensor._rmatmul(other, self)


    def __matmul__(self, other: Tensor | "MonarchTensor") -> Tensor: # type: ignore
        return MonarchTensor._matmul(self, other)


    @staticmethod
    def _matmul(monarch: MonarchTensor | nn.Module, other: Tensor | TensorLike) -> Tensor | NotImplementedError:
        match other:
            case TensorLike():
                return MonarchTensor._matmul(monarch, other.dense)
            
            case Tensor():
                if other.shape[0] != monarch.n:
                    raise ValueError(f"Dimension mismatch: MonarchTensor matmul requires Tensor dim {monarch.n}, but got {other.shape[-1]}")
                
                return monarch.P2 @ (monarch.L @ (monarch.P1 @ (monarch.R @ other)))
            
            case _:
                return NotImplemented


    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, monarch: MonarchTensor | nn.Module) -> Tensor | NotImplementedError:
        match other:
            case TensorLike():
                return MonarchTensor._rmatmul(other.dense, monarch)
            
            case Tensor():
                if other.shape[-1] != monarch.n:
                    raise ValueError(f"Dimension mismatch: MonarchTensor rmatmul requires Tensor dim {monarch.n}, but got {other.shape[-1]}")
                
                return other @ monarch.P2 @ monarch.L @ monarch.P1 @ monarch.R
            
            case _:
                return NotImplemented
            
    
    def to(self, *args, **kwargs) -> "MonarchTensor":
        data_on_device = torch.Tensor.to(self, *args, **kwargs)
        instance = data_on_device.as_subclass(MonarchTensor)
        instance.P1 = self.P1.to(*args, **kwargs)
        instance.P2 = self.P2.to(*args, **kwargs)
        instance.m = self.m
        instance.n = self.n
        return instance



    @staticmethod
    def random(m: int) -> "MonarchTensor":
        R = torch.randn(m, m, m)
        L = torch.randn(m, m, m)

        W = torch.full((2, m, m, m), torch.nan)
        W[0, ...] = R
        W[1, ...] = L

        return MonarchTensor(
            torch.stack([R, L])
        )
    
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cloned_tensor_data = torch.Tensor.clone(self).detach()
        
        new_instance = type(self)(cloned_tensor_data)

        memo[id(self)] = new_instance

        new_instance.P1 = copy.deepcopy(self.P1, memo)
        new_instance.P2 = copy.deepcopy(self.P2, memo)
        
        new_instance.m = self.m
        new_instance.n = self.n
        
        return new_instance
