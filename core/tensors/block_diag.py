from __future__ import annotations
import torch
from torch import Tensor, nn

from einops import rearrange, einsum # type: ignore

from ..tensors.tensorlike import TensorLike


class BlockDiagTensor(TensorLike):
    def __new__(cls, data: Tensor, *args, **kwargs): # type: ignore
        if not isinstance(data, Tensor): # type: ignore
            raise TypeError("BlockDiagTensor must be initialized with a Tensor")
        if data.ndim != 3:
            raise ValueError("BlockDiagTensor must have 3 dimensions")
        return data.as_subclass(cls)
    

    def __init__(self, data: Tensor, *args, **kwargs): # type: ignore
        self.nblocks, self.b2, self.b1 = self.shape

    @property
    def dense(self) -> Tensor:
        blocks = torch.unbind(self, dim=0)
        return torch.block_diag(*blocks).as_subclass(Tensor) # type: ignore


    def __matmul__(self, other: Tensor | TensorLike) -> Tensor: # type: ignore
        return self._matmul(self, other) # type: ignore
    
    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor: # type: ignore
        return self._rmatmul(other, self) # type: ignore
    


    @staticmethod
    def _matmul(tensor : BlockDiagTensor | nn.Parameter, other: Tensor | TensorLike) -> Tensor | NotImplementedError:
        nblocks, _, b1 = tensor.shape

        match other:
            case TensorLike():
                return tensor._matmul(tensor, other.dense) # type: ignore
            
            case Tensor():

                if other.ndim == 1:                    
                    if other.shape[0] != nblocks*b1:
                        raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {nblocks*b1}, but got {other.shape[0]}")

                    other = rearrange(other, "(nblocks b1) -> nblocks b1", nblocks=nblocks, b1=b1)
                    mul = einsum(tensor.as_subclass(Tensor), other, "nblocks b2 b1, nblocks b1 -> nblocks b2")
                    return rearrange(mul, "nblocks b2 -> (nblocks b2)") # type: ignore
                
                if other.ndim == 2:
                    if other.shape[0] != nblocks*b1:
                        raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {nblocks*b1}, but got {other.shape[0]}")

                    other = rearrange(other, "(nblocks b1) B -> nblocks b1 B", nblocks=nblocks, b1=b1)
                    mul = einsum(tensor.as_subclass(Tensor), other, "nblocks b2 b1, nblocks b1 B -> nblocks b2 B")
                    return rearrange(mul, "nblocks b2 B -> (nblocks b2) B") # type: ignore
                
                return NotImplemented
            

    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, tensor: BlockDiagTensor | nn.Parameter) -> Tensor | NotImplementedError:
        nblocks, b2, _ = tensor.shape

        match other:
            case TensorLike():
                return tensor._rmatmul(other.dense, tensor) # type: ignore
            
            case Tensor():
                if other.ndim == 1:
                    if other.shape[0] != nblocks*b2:
                        raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {nblocks*b2}, but got {other.shape[0]}")

                    other = rearrange(other, "(nblocks b2) -> nblocks b2", nblocks=nblocks, b2=b2)
                    mul = einsum(tensor.as_subclass(Tensor), other, "nblocks b2 b1, nblocks b2 -> nblocks b1")
                    return rearrange(mul, "nblocks b1 -> (nblocks b1)") # type: ignore
                
                if  other.ndim == 2:
                    if other.shape[1] != nblocks*b2:
                        raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {nblocks*b2}, but got {other.shape[1]}")
                
                    other = rearrange(other, "B (nblocks b2) -> B nblocks b2", nblocks=nblocks, b2=b2)
                    mul = einsum(tensor.as_subclass(Tensor), other, "nblocks b2 b1, B nblocks b2 -> B nblocks b1")
                    return rearrange(mul, "B nblocks b1 -> B (nblocks b1)") # type: ignore
                
                return NotImplemented            

    @property      
    def T(self) -> "BlockDiagTensor": # type: ignore
        return einsum(self, "nblocks b2 b1 -> nblocks b1 b2")
    

    @staticmethod    
    def random(nblocks: int, b1: int, b2: int) -> "BlockDiagTensor":
        return BlockDiagTensor(torch.randn(nblocks, b2, b1))

    @staticmethod
    def zeros(nblocks: int, b1: int, b2: int) -> "BlockDiagTensor":
        return BlockDiagTensor(torch.zeros(nblocks, b2, b1))

    def to(self, *args, **kwargs) -> "BlockDiagTensor":
        data_on_device = torch.Tensor.to(self, *args, **kwargs)
        return BlockDiagTensor(data_on_device)
    

    def __deepcopy__(self, memo: dict[int, "BlockDiagTensor"]) -> "BlockDiagTensor":
        if id(self) in memo:
            return memo[id(self)]
        
        cloned_tensor_data = torch.Tensor.clone(self).detach()
        new_instance = type(self)(cloned_tensor_data)
        memo[id(self)] = new_instance

        return new_instance