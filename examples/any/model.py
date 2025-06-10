from core.models.pinn import PINN, DLinear
from torch import Tensor
from collections.abc import Iterable, Callable
from torch import nn


class AnyPINN(PINN):
    def __init__(self,  layers: Iterable[DLinear],
                        pde: Callable[[PINN, Tensor, Tensor], Tensor],
                        activation: type[nn.Module]=nn.Tanh, 
                        lmda:float =1.0) -> None:
        super().__init__(layers, activation, lmda)

        self.pde_func = pde


    def pde(self, a: Tensor, u: Tensor) -> Tensor:
        return self.pde_func(self, a, u)
        
    
    