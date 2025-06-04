import torch
from torch import Tensor, nn
from ..layers.parameterlike import ParameterLike
from ..layers.block_diag import BlockDiagParameter
from ..tensors.tensorlike import Tensorable
from ..layers.learnable_permutation import LearnablePermutation

from ..tensors.steam import STEAMTensor
import torch.nn.functional as F


import math


class STEAMParameter(ParameterLike):
    def __init__(self, tensor: STEAMTensor):
        super().__init__() # type: ignore

        self.P0 = LearnablePermutation(tensor.P0)
        self.R = BlockDiagParameter(tensor.R)
        Pbar = tensor.Pbar
        self.L = BlockDiagParameter(tensor.L)
        self.P2 = LearnablePermutation(tensor.P2)


        self.m = tensor.shape[1]
        self.n = self.m**2

        self.register_buffer("Pbar", Pbar)

    
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] != self.n:
            raise ValueError(f"Input tensor must have shape {self.n}, but got {x.shape[0]}")
        
        return STEAMTensor._matmul(self, x)
    

    def __matmul__(self, x: Tensor | Tensorable) -> Tensor:
        match x:
            case Tensorable():
                return self.__matmul__(x.dense)
            
            case Tensor():
                return self.forward(x)
            
            case _:
                return NotImplemented
            
    @property
    def dense(self) -> Tensor:
        return self.P2.dense @ (self.L @ (self.Pbar @ (self.R @ (self.P0.dense))))
    



class STEAMLinear(STEAMParameter):
    def __init__(self, in_features: int, out_features: int, bias: Tensor | bool = True):
        
        m = math.ceil(math.sqrt(max(in_features, out_features)))

        super().__init__(STEAMTensor.random(m)) # type: ignore
        
        
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
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        y = super().forward(x.T)
        return self.postprocess(y.T) + self.bias
    

    @staticmethod
    def from_tensors(steam : STEAMParameter | STEAMTensor, bias : Tensor):
        layer = STEAMLinear(steam.n, steam.n, bias)
        
        if isinstance(steam, STEAMTensor):
            #layer.P0.X.weight = steam.P0
            #layer.P2.X.weight = steam.P2
#
            #layer.R.block_diag.data = steam.R.clone().detach()
            #layer.L.block_diag.data = steam.L.clone().detach()


            layer.P0 = LearnablePermutation(steam.P0)
            layer.P2 = LearnablePermutation(steam.P2)

            layer.R = BlockDiagParameter(steam.R)
            layer.L = BlockDiagParameter(steam.L)
            
            layer.Pbar = steam.Pbar 

        elif isinstance(steam, STEAMParameter):
            layer.P0.X = steam.P0.X.clone().detach()
            layer.P2.X = steam.P2.X.clone().detach()

            layer.R.block_diag.data = steam.R.block_diag.data.clone().detach()
            layer.L.block_diag.data = steam.L.block_diag.data.clone().detach()
            layer.Pbar = steam.Pbar


        return layer