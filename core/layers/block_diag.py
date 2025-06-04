from torch import nn, Tensor
from einops import rearrange, einsum # type: ignore
import torch


from ..tensors.block_diag import BlockDiagTensor
from ..tensors.tensorlike import TensorLike

from ..layers.parameterlike import ParameterLike



class BlockDiagParameter(ParameterLike):

    def __init__(self, block_diag: BlockDiagTensor):
        super().__init__() # type: ignore

        self.block_diag = nn.Parameter(block_diag.as_subclass(Tensor))

        self.nblocks = block_diag.shape[0]
        self.b2 = block_diag.shape[1]
        self.b1 = block_diag.shape[2]

        self.in_features = self.nblocks * self.b1
        self.out_features = self.nblocks * self.b2


    def forward(self, x: Tensor | TensorLike, right: bool = False) -> Tensor:
        match right:
            case True:
                return BlockDiagTensor._rmatmul(x, self.block_diag) # type: ignore
            case False:
                return BlockDiagTensor._matmul(self.block_diag, x) # type: ignore
            
    
    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self.forward(other, right=False)
    

    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self.forward(other, right=True)
    

    @property
    def dense(self) -> Tensor:
        return BlockDiagTensor(self.block_diag.data).dense
    

    
