from __future__ import annotations
from ..tensors.tensorlike import TensorLike
from ..tensors.block_diag import BlockDiagTensor
from ..tensors.permutation import bit_rev, BitRevPermutationTensor

from ..layers.learnable_permutation import PermutationSTE

from ..utils.butterfly import blockdiag_butterfly_project

from torch import Tensor, nn
import torch
from typing import Any # Added for type hinting
import copy

import math

class STEAMTensor(TensorLike):
    def __new__(cls, tensor: Tensor, permutations: Tensor,*args, **kwargs): # type: ignore
        if not isinstance(tensor, Tensor): # type: ignore
            raise TypeError("STEAMTensor must be initialized with a Tensor")
        if tensor.ndim != 4:
            raise ValueError("STEAMTensor must have 4 dimensions")
        
        if tensor.shape[1] != tensor.shape[2] != tensor.shape[3]:
            raise ValueError("STEAMTensor must have 4 dimensions with the second and third dimensions equal")

        return tensor.as_subclass(cls)
    


    def __init__(self, tensor: Tensor, permutations: Tensor, *args, **kwargs): # type: ignore
        self.m: int = tensor.shape[1]
        self.n: int = self.m**2

        self.permutations = permutations
        self.Pbar = BitRevPermutationTensor(self.n)

    
    @property 
    def L(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[1, ...])

    
    
    @property 
    def R(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[0, ...])
    

    @property
    def P0(self) -> Tensor:
        return self.permutations[0, ...]
    
    @property
    def P2(self) -> Tensor:
        return self.permutations[1, ...]
    

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


    @P0.setter
    def P0(self, new_P0: Tensor):
        if new_P0.shape != (self.n, self.n):
            raise ValueError(f"Dimension mismatch: P0 must have shape ({self.n}, {self.n}), but got {new_P0.shape}")
        self.permutations[0, ...] = new_P0.clone().detach().data

    
    @P2.setter
    def P2(self, new_P2: Tensor):
        if new_P2.shape != (self.n, self.n):
            raise ValueError(f"Dimension mismatch: P2 must have shape ({self.n}, {self.n}), but got {new_P2.shape}")
        self.permutations[1, ...] = new_P2.clone().detach().data


    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return STEAMTensor._rmatmul(other, self)
    
    def __matmul__(self, other: Tensor | STEAMTensor) -> Tensor: # type: ignore
        return STEAMTensor._matmul(self, other)


    @staticmethod
    def _matmul(steam: 'STEAMTensor | nn.Module' , other: Tensor | TensorLike) -> Tensor:
        match other:
            case TensorLike():
                return STEAMTensor._matmul(steam, other.dense)
            
            case Tensor():
                if other.shape[0] != steam.n:
                    raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {steam.n}, but got {other.shape[0]}")
                return steam.P2 @ (steam.L @ (steam.Pbar @ (steam.R @ (steam.P0 @ other))))
            
            case _:
                return NotImplemented
            

    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, steam: 'STEAMTensor | nn.Module') -> Tensor:
        match other:
            case TensorLike():
                return STEAMTensor._rmatmul(other.dense, steam)
            
            case Tensor():
                if other.shape[-1] != steam.n:
                    raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {steam.n}, but got {other.shape[0]}")
                return other @ steam.P2 @ steam.L @ steam.Pbar @ steam.R @ steam.P0
            
            case _:
                return NotImplemented
            


    @staticmethod
    def random(m: int) -> "STEAMTensor":
        n = m*m
        R = torch.randn(m, m, m)
        L = torch.randn(m, m, m)

        P0 = PermutationSTE.apply(torch.randn(n, n)) # type: ignore
        P2 = PermutationSTE.apply(torch.randn(n, n)) # type: ignore
        return STEAMTensor(torch.stack([R, L]), torch.stack([P0, P2])) # type: ignore
    

    @property
    def dense(self)-> Tensor:
        return self.P2 @ (self.L @ (self.Pbar @ (self.R @ self.P0)))
    
    def to(self, *args: Any, **kwargs: Any) -> "STEAMTensor":
        data_on_device = torch.Tensor.to(self, *args, **kwargs)
        instance = data_on_device.as_subclass(STEAMTensor)
        instance.permutations = self.permutations.to(*args, **kwargs)
        instance.Pbar = self.Pbar.to(*args, **kwargs)
        instance.m = self.m
        instance.n = self.n
        return instance
    

    @staticmethod
    def from_dense(A: Tensor, T: int = 100, alpha: float = 0.001)-> "STEAMTensor":
        if A.ndim != 2:
            raise ValueError("Tensor must be 2D")
        
        n = A.shape[0]
        if ( m:=math.isqrt(n) )**2 != n :
            raise ValueError(f"Input tensor must be perfect square, but got shape {A.shape}")

        device = A.device # Get device from input tensor A
        Pbar = bit_rev(n).dense.to(device)
        P0 = bit_rev(n).dense.to(device)
        P2 = bit_rev(n).dense.to(device)
        X0 = bit_rev(n).dense.to(device)
        X2 = bit_rev(n).dense.to(device)

        R, L = blockdiag_butterfly_project(A @ P0.T)
        R = R.to(device) # Ensure R is on the correct device
        L = L.to(device) # Ensure L is on the correct device


        L_best, R_best, P0_best, P2_best = L.clone(), R.clone(), P0.clone(), P2.clone() # Initialize with initial tensors
        norm_best = torch.norm( (P2 @ BlockDiagTensor(L).dense @  Pbar @ BlockDiagTensor(R).dense @ P0) - A).item()/torch.norm(A)
        # norm_best = float("inf") # Previous initialization


        for t in range(T):
            #print("t = ", t)
            # Ensure operands for torch.norm are on the same device
            L_dense_Pbar_R_dense = BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense
            norm_val_squared = torch.norm(L_dense_Pbar_R_dense)**2
            
            if norm_val_squared.item() == 0: # Avoid division by zero
                nu = torch.tensor(float('inf'), device=device) # Or handle differently
            else:
                nu = 1 / (alpha * norm_val_squared)

            # git veut pas marcher du coup je crée un version avec uniquement ce texte

            # Ensure all parts of the gradient calculation are on the correct device
            term1_X0 = (P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense).T
            term2_X0 = ((P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0) - A)
            X0_update = term1_X0 @ term2_X0
            X0 = X0 - nu * X0_update


            term1_X2 = ( (P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0) - A)
            term2_X2 = (BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0).T
            X2_update = term1_X2 @ term2_X2
            X2 = X2 - nu * X2_update


            P0 = PermutationSTE.apply(X0)
            P2 = PermutationSTE.apply(X2)

            # Ensure results of blockdiag_butterfly_project are on the correct device
            projection_input = Pbar @ P2.T @ A @ P0.T
            R_new, L_new = blockdiag_butterfly_project(projection_input)
            R, L = R_new.to(device), L_new.to(device)

            
            current_norm_tensor = (P2 @ BlockDiagTensor(L).dense @  Pbar @ BlockDiagTensor(R).dense @ P0) - A
            current_norm = (torch.norm(current_norm_tensor).item() / torch.norm(A)).item() if torch.norm(A).item() != 0 else float('inf')

            if current_norm < norm_best:
                norm_best = current_norm
                L_best = L.clone()
                R_best = R.clone()
                P0_best = P0.clone()
                P2_best = P2.clone()

                # print("current norm =", current_norm, "best norm =", norm_best)

        # Ensure final tensors for STEAMTensor are on the correct device.
        # stack will use the device of the input tensors, which should now be correct.
        return STEAMTensor(torch.stack([R_best, L_best]), torch.stack([P0_best, P2_best]))

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        cloned_tensor_data = torch.Tensor.clone(self).detach()
        cloned_permutations = copy.deepcopy(self.permutations, memo)
        
        new_instance = type(self)(cloned_tensor_data, cloned_permutations)
        
        memo[id(self)] = new_instance
        
        new_instance.Pbar = copy.deepcopy(self.Pbar, memo)
        new_instance.m = self.m
        new_instance.n = self.n
        
        return new_instance
