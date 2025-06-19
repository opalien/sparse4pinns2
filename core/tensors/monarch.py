from __future__ import annotations
from ..tensors.tensorlike import TensorLike
from ..tensors.block_diag import BlockDiagTensor
from ..tensors.permutation import bit_rev, BitRevPermutationTensor

from torch import Tensor, nn
import torch
import copy

from ..utils.butterfly import blockdiag_butterfly_project, BlockdiagButterflyMultiply


from  einops import rearrange, einsum # type: ignore


_fast_path_warning_shown: bool = False


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

                #use_fast_path = (
                #    other.is_cuda and
                #    other.dtype in [torch.float32, torch.float16]
                #)

                if True:#use_fast_path:
                    try:
                        global _fast_path_warning_shown
                        x = other

                        input_was_1d = (x.ndim == 1)
                        if input_was_1d:
                            x = x.unsqueeze(0)

                        if hasattr(monarch.R, 'block_diag'):
                            R_tensor = monarch.R.block_diag
                        else:
                            R_tensor = monarch.R

                        if hasattr(monarch.L, 'block_diag'):
                            L_tensor = monarch.L.block_diag
                        else:
                            L_tensor = monarch.L

                        x_t = x.transpose(0, 1)

                        print(f"MonarchTensor fast path: x_t.device={x_t.device}, "
                              f"R_tensor.device={R_tensor.device}, "
                              f"L_tensor.device={L_tensor.device}")

                        output_t = BlockdiagButterflyMultiply.apply(x_t, R_tensor, L_tensor)
                        
                        output = output_t.transpose(0, 1)

                        if input_was_1d:
                            output = output.squeeze(0)
                        
                        return output


                    except Exception as e:
                        if not _fast_path_warning_shown:
                            _fast_path_warning_shown = True
                                
                            print(f"Fast Monarch multiply failed with error: {e}. "
                                        f"Falling back to the slow implementation.")
                        pass

                #return monarch.P2.to(other.device) @ (monarch.L.to(other.device) @ (monarch.P1.to(other.device) @ (monarch.R.to(other.device) @ other)))
                #return  (monarch.L.to(other.device) @ (monarch.R.to(other.device) @ other))


            case _:
                return NotImplemented


#    @staticmethod
#    def _matmul(monarch: MonarchTensor | nn.Module, other: Tensor | TensorLike) -> Tensor | NotImplementedError:
#        match other:
#            case TensorLike():
#                return MonarchTensor._matmul(monarch, other.dense)
#            
#            case Tensor():
#                if other.shape[0] != monarch.n:
#                    raise ValueError(f"Dimension mismatch: MonarchTensor matmul requires Tensor dim {monarch.n}, but got {other.shape[-1]}")
#                
#
#                return monarch.P2 @ (monarch.L @ (monarch.P1 @ (monarch.R @ other)))
#            
#            case _:
#                return NotImplemented


    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, monarch: MonarchTensor | nn.Module) -> Tensor | NotImplementedError:
        raise NotImplementedError("MonarchTensor does not support right matmul with TensorLike. Use MonarchTensor._rmatmul instead.")
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
