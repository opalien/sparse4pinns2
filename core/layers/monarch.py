import torch
from torch import Tensor, nn

from ..tensors.monarch import MonarchTensor
from ..layers.parameterlike import ParameterLike
from ..layers.block_diag import BlockDiagParameter
from ..tensors.permutation import bit_rev, BitRevPermutationTensor
from ..tensors.tensorlike import Tensorable

import torch.nn.functional as F

from ..layers.parameterlike import ParameterLike


import math


class MonarchParameter(ParameterLike):
    
    def __init__(self, monarch: MonarchTensor):
        super().__init__()

        self.m = monarch.m
        self.n = monarch.n
        self.R = BlockDiagParameter(monarch.R)
        P1 = BitRevPermutationTensor(self.n)
        self.L = BlockDiagParameter(monarch.L)
        P2  = BitRevPermutationTensor(self.n)

        self.register_buffer("P1", P1)
        self.register_buffer("P2", P2)

    
    def __matmul__(self, x: Tensor | Tensorable) -> Tensor:
        return self.forward(x, right=False)


    def __rmatmul__(self, x: Tensor | Tensorable) -> Tensor:
        return self.forward(x, right=True)
    


    def forward(self, x: Tensor, right: bool = False) -> Tensor:
        match right:
            case True:
                return MonarchTensor._rmatmul(x, self)
            case False:
                return MonarchTensor._matmul(self, x)
            

    def tensor(self) -> MonarchTensor:
        return MonarchTensor(torch.stack([self.R.block_diag.data, self.L.block_diag.data]))
    
    @property
    def dense(self) -> Tensor:
        return self.tensor().dense
    



class MonarchLinear(MonarchParameter):
    def __init__(self, in_features: int, out_features: int, bias: Tensor | bool = True):
        
        m = math.ceil(math.sqrt(max(in_features, out_features)))
        super().__init__(MonarchTensor.random(m))


        match bias:
            case True:
                self.bias = nn.Parameter(torch.zeros(out_features))
            case False:
                self.bias = torch.zeros(out_features)
            case Tensor():
                if bias.shape != (out_features,):
                    raise ValueError(f"Bias must have shape ({out_features},), but got {bias.shape}")
                self.bias = nn.Parameter(bias)
            case _:
                raise ValueError(f"Bias must be a Tensor, bool, but got {type(bias)}")
            
        

        self.in_features = in_features
        self.out_features = out_features

    
    def preprocess(self, x:Tensor):
        in_features = x.shape[-1]
        if in_features < self.n:
            x = F.pad(x, (0, self.n - in_features))
        return x
    

    def postprocess(self, output:Tensor):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output
    

    def forward(self, x: Tensor, right=False) -> Tensor:
        x = self.preprocess(x)

        y = super().forward(x.T, right=right)
        return self.postprocess(y.T) + self.bias
    


    @staticmethod
    def from_tensors(monarch : MonarchParameter | MonarchTensor, bias : Tensor):
        layer = MonarchLinear(monarch.n, monarch.n, bias)


        if isinstance(monarch, MonarchTensor):
            layer.R.block_diag.data = monarch.R.as_subclass(Tensor).clone().detach()
            layer.L.block_diag.data = monarch.L.as_subclass(Tensor).clone().detach()

        elif isinstance(monarch, MonarchParameter):
            layer.R.block_diag.data = monarch.R.block_diag.data.clone().detach()
            layer.L.block_diag.data = monarch.L.block_diag.data.clone().detach()
        else:
            raise TypeError(f"Expected MonarchTensor or MonarchParameter, got {type(monarch)}")

        return layer
        